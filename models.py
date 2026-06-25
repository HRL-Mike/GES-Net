import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


# 1. Base TCN layer with dilated receptive fields
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size=3):
        super(DilatedResidualLayer, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size,
                                      padding=padding, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


# 2. Gated cross-attention block
class GatedCrossAttention(nn.Module):
    def __init__(self, d_model=64):
        super(GatedCrossAttention, self).__init__()
        self.gate_q = nn.Linear(d_model, d_model)
        self.gate_v = nn.Linear(d_model, d_model)
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, text_q, visual_kv):
        B, T_v, C = visual_kv.shape
        Q = self.q_linear(text_q)
        K = self.k_linear(visual_kv)
        V = self.v_linear(visual_kv)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(C)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        # Query-dependent gate for semantic sparsity.
        gate = torch.sigmoid(self.gate_q(text_q) + self.gate_v(context))
        context = context * gate
        return self.out_proj(context), gate


# 3. CoT semantic decomposition block
class CoTBranch(nn.Module):
    def __init__(self, visual_dim=768, text_dim=512, d_model=64):
        super(CoTBranch, self).__init__()
        # TDF doubles the visual dimension when enabled.
        self.v_proj = nn.Linear(visual_dim, d_model)
        self.t_proj = nn.Linear(text_dim, d_model)

        self.cross_attn1 = GatedCrossAttention(d_model)
        self.cross_attn2 = GatedCrossAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x_v, x_t):
        v_feat = self.v_proj(x_v)
        t_feat = self.t_proj(x_t)

        out, _ = self.cross_attn1(t_feat, v_feat)
        out = self.norm1(out + t_feat)

        raw_out2, gate2 = self.cross_attn2(out, v_feat)
        final_out2 = self.norm2(raw_out2 + out)
        return final_out2, raw_out2, gate2


# 4. Single-stage TCN block
class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes_g, num_classes_e, k_size=3):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, kernel_size=k_size))
            for i in range(num_layers)
        ])
        self.conv_out_gest = nn.Conv1d(num_f_maps, num_classes_g, 1)
        self.conv_out_err = nn.Conv1d(num_f_maps, num_classes_e, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        return out, self.conv_out_gest(out), self.conv_out_err(out)


# 5. Semantic-driven main model
class GESNet(nn.Module):
    def __init__(self, clip_feat_path, num_actions=7, num_stages=4, num_layers=10, use_soft_error=False, use_tdf=True):
        super(GESNet, self).__init__()
        self.num_actions = num_actions
        self.num_observers = num_actions + 1  # Actions plus one global observer.
        self.use_soft_error = use_soft_error
        self.use_tdf = use_tdf

        # CLIP action prompts plus one global motion observer.
        self.register_buffer("clip_base", torch.from_numpy(np.load(clip_feat_path)).float())
        self.learnable_delta = nn.Parameter(torch.randn(num_actions, 512) * 0.01)
        self.g_global_prompt = nn.Parameter(torch.randn(1, 512) * 0.02)

        # CoT branch uses doubled visual features when TDF is enabled.
        self.v_dim = 768 * 2 if use_tdf else 768
        self.cot = CoTBranch(visual_dim=self.v_dim, text_dim=512, d_model=64)

        # Produces competitive gate weights: [B*T, 16, 768].
        self.action_gate_projs = nn.Linear(64, 768)
        # Auxiliary observer-alignment supervision.
        self.cot_classifier = nn.Linear(64, self.num_observers)
        self.semantic_dim = self.num_observers * 64

        kernels = [1, 3, 5, 7]
        self.stage1 = SingleStageModel(num_layers, 64, self.semantic_dim, num_actions, 2, k_size=kernels[0])
        self.stages = nn.ModuleList([
            copy.deepcopy(SingleStageModel(num_layers, 64, 64, num_actions, 2, k_size=kernels[i + 1]))
            for i in range(num_stages - 1)
        ])

        self.regression_head = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, dv3_feat, g_labels, lengths, tau=0.1, training_mode=True):
        B, T_max, C_v = dv3_feat.shape
        window_size = 61
        pad = window_size // 2

        if training_mode:
            dv3_feat = F.dropout(dv3_feat, p=0.1, training=True)

        # --- A. TDF motion extraction from CLS tokens ---
        # Capture temporal change to reduce static-background noise.
        if self.use_tdf:
            k_dist = 5
            # Difference from five frames earlier as a motion signal.
            cls_padded = F.pad(dv3_feat.transpose(1, 2), (k_dist, 0), mode='replicate').transpose(1, 2)
            tdf_signal = dv3_feat - cls_padded[:, :T_max, :]
            v_pool = torch.cat([dv3_feat, tdf_signal], dim=-1)
        else:
            v_pool = dv3_feat

        # --- B. Action semantic decomposition ---
        prompts = torch.cat([self.clip_base + self.learnable_delta, self.g_global_prompt], dim=0)
        prompts = prompts.unsqueeze(0).expand(B, -1, -1)

        # Prepare sliding windows.
        x_v = v_pool.transpose(1, 2)
        x_v_padded = F.pad(x_v, (pad, pad), mode='replicate')
        windows = x_v_padded.unfold(2, window_size, 1)  # [B, 1536, window_size, T]
        windows = windows.permute(0, 3, 2, 1).contiguous().view(B * T_max, window_size, -1)

        prompts_expanded = prompts.repeat_interleave(T_max, dim=0)

        # Main interaction features: cot_stream_flat is [B*T, 16, 64].
        cot_stream_flat, _, _ = self.cot(windows, prompts_expanded)
        cot_stream_reshaped = cot_stream_flat.view(-1, self.num_observers, 64)

        # --- C. Competitive gating ---
        raw_action_gates = self.action_gate_projs(cot_stream_reshaped)
        action_weights = F.softmax(raw_action_gates / tau, dim=1)

        # 2. Element-wise scheduled sampling guide for training.
        if training_mode and g_labels is not None:
            B_T = B * T_max
            # Ones mean the model uses its own prediction.
            gt_mask = torch.ones(B_T, self.num_observers, 1).to(dv3_feat.device)
            # Frames with labels >= 0 are guided.
            guide_indices = g_labels.view(-1, 1, 1)
            is_guided = (guide_indices >= 0)  # Teacher forcing excludes -100 labels.

            if is_guided.any():
                # Keep only the GT channel and global observer for guided frames.
                forced_mask = torch.zeros_like(gt_mask)
                safe_indices = guide_indices.clone()
                safe_indices[~is_guided] = 0  # Avoid scatter out-of-range errors.
                forced_mask.scatter_(1, safe_indices.long(), 1.0)
                forced_mask[:, self.num_observers - 1, :] = 1.0
                # Use forced masks for guided frames, otherwise keep all ones.
                gt_mask = torch.where(is_guided.expand(-1, self.num_observers, -1), forced_mask, gt_mask)

            action_weights = action_weights * gt_mask
            action_weights = action_weights / (action_weights.sum(dim=1, keepdim=True) + 1e-8)

        # 3. Global visual gate for visualization and diagnostics: [B, T, 768].
        # Retain channels selected by the strongest observer per frame.
        v_gate = action_weights.sum(dim=1).view(B, T_max, 768)
        # 4. Alignment supervision logits: [B*T, 16, 16].
        cot_logits = self.cot_classifier(cot_stream_reshaped)

        # --- D. Semantic stream construction ---
        # Apply competitive weights to sharpen observer energy.
        obs_importance = action_weights.mean(dim=-1, keepdim=True)  # [B*T, 16, 1]
        sparse_stream = cot_stream_reshaped * obs_importance  # Enforce semantic sparsity.

        cot_stream = sparse_stream.view(B, T_max, -1)  # [B, T, 1024]
        raw_feat = sparse_stream.view(B, T_max, self.num_observers, -1)  # For radar plots.

        # --- E. MS-TCN temporal modeling ---
        x = cot_stream.transpose(1, 2)
        out, out_g, out_e = self.stage1(x)
        gest_preds, err_preds = [out_g], [out_e]
        for stage in self.stages:
            out, out_g, out_e = stage(out)
            gest_preds.append(out_g)
            err_preds.append(out_e)

        # --- F. Aggregation and regression ---
        mask = torch.arange(T_max).expand(B, T_max).to(dv3_feat.device)
        mask = (mask < lengths.unsqueeze(1)).float().unsqueeze(1)

        if self.use_soft_error:
            err_probs = F.softmax(err_preds[-1], dim=1)[:, 1:, :]
            soft_weights = (1.0 + err_probs) * mask
            v_global = torch.sum(out * soft_weights, dim=-1) / torch.sum(soft_weights, dim=-1)
        else:
            v_global = torch.sum(out * mask, dim=-1) / lengths.unsqueeze(1).float()

        # 2. Temporal jitter feature.
        v_diff = (out - v_global.unsqueeze(-1)) * mask
        v_jitter = torch.sqrt(torch.sum(v_diff ** 2, dim=-1) / (lengths.unsqueeze(1).float() + 1e-6))
        # 3. Concatenate features for the regression head.
        v_combined = torch.cat([v_global, v_jitter], dim=-1)
        final_score = self.regression_head(v_combined)

        return {
            "score": final_score.squeeze(-1),
            "gestures": gest_preds,
            "errors": err_preds,
            "cot_stream": cot_stream,
            "cot_gates": action_weights.view(B, T_max, 16, -1),
            "raw_feat": raw_feat,
            "cot_logits": cot_logits,
            "v_gate": v_gate,
            "global_energy": obs_importance[:, self.num_observers - 1, :].mean(),
            "v_jitter": v_jitter
        }
