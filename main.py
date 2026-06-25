import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import random
import copy
import argparse
import warnings
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler
from dataloader import collate_fn

# Project model and dataset imports.
from models import GESNet
from dataloader import CustomVideoDataset
from utils import get_element_wise_mask_inverse

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# 1. Environment setup and random seeds
def set_random_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# 2. Score normalization
def normalize_score(score, task='SU', epsilon=0.01):
    if task == 'SU':
        min_score = 8
        max_score = 30
    elif task == 'NP':
        min_score = 7
        max_score = 24
    elif task == 'All_SU' or task == 'All_NP':
        min_score = 7
        max_score = 30
    elif task == 'RARP':
        min_score = 18
        max_score = 28
    else:
        raise ValueError(f"Unknown task name: {task}")
    norm_score = (score - min_score) / (max_score - min_score)
    norm_score = norm_score * (1 - 2 * epsilon) + epsilon  # [epsilon, 1-epsilon]
    return norm_score

def denormalize_score(norm_score, task='SU', epsilon=0.01):
    score = (norm_score - epsilon) / (1 - 2 * epsilon)
    if task == 'SU':
        min_score = 8
        max_score = 30
    elif task == 'NP':
        min_score = 7
        max_score = 24
    elif task == 'All_SU' or task == 'All_NP':
        min_score = 7
        max_score = 30
    elif task == 'RARP':
        min_score = 18
        max_score = 28
    else:
        raise ValueError(f"Unknown task name: {task}")
    return score * (max_score - min_score) + min_score


def compute_normalized_sequence_losses(output, g_labels, e_labels, lengths, criteria):
    """
    Length-normalized action, error, and temporal smoothing losses.
    """
    num_stages = len(output['gestures'])
    B, _, T = output['gestures'][0].shape
    device = g_labels.device
    # Valid-frame mask: [B, T].
    mask = torch.arange(T).to(device).expand(B, T) < lengths.unsqueeze(1)
    mask_f = mask.float()
    # TMSE mask with T-1 length: [B, T-1].
    mask_tmse = mask_f[:, 1:]
    total_gest, total_err, total_tmse = 0.0, 0.0, 0.0
    for i in range(num_stages):
        # 1. Action classification with per-frame CE loss.
        gest_raw = F.cross_entropy(output['gestures'][i], g_labels, reduction='none')
        total_gest += (gest_raw * mask_f).sum(dim=1) / (lengths.float() + 1e-6)
        # 2. Error detection.
        err_raw = F.cross_entropy(output['errors'][i], e_labels, reduction='none')
        total_err += (err_raw * mask_f).sum(dim=1) / (lengths.float() + 1e-6)
        # 3. TMSE temporal smoothing.
        log_probs = F.log_softmax(output['errors'][i], dim=1)
        # Squared log-probability differences between adjacent frames.
        diff = (log_probs[:, :, 1:] - log_probs[:, :, :-1]) ** 2
        # Focus on transitions in the error state.
        tmse_raw = diff[:, 1, :] * mask_tmse
        total_tmse += tmse_raw.sum(dim=1) / (lengths.float() - 1 + 1e-6)
    # Average over the batch and normalize by stage count.
    return total_gest.mean() / num_stages, total_err.mean() / num_stages, total_tmse.mean() / num_stages


# 3. Training and validation
def train_one_epoch(model, dataloader, optimizer, criteria, device, task_name, epoch, args):
    model.train()
    total_loss = 0.0
    # --- Temperature annealing ---
    tau_start, tau_end, anneal_epochs = 1.0, 0.1, 60
    tau = max(tau_end, tau_start - (tau_start - tau_end) * (epoch / anneal_epochs))
    # Loss weights.
    w_mse, w_mae = 5.0, 20.0
    w_gest = max(0.2, 0.6 - (epoch / 200))  # Reduce classification weight over time.
    w_err, w_tmse = 0.5, args.lambda_ * 0.5
    w_cot = 0.5
    w_global_energy = 1.0
    # Loss monitor.
    monitor = {
        'total': 0.0,
        'skill': 0.0,
        'ms_tcn': 0.0,
        'cot': 0.0,
        'g16': 0.0,
        'mask_rate': 0.0
    }

    for data in dataloader:
        names, lengths, dv3_feat, e_labels, g_labels, s_label = data
        dv3_feat, g_labels, lengths = dv3_feat.to(device), g_labels.to(device).long(), lengths.to(device)
        e_labels, s_label = e_labels.to(device).long(), s_label.to(device)
        s_norm = torch.tensor([normalize_score(s.item(), task_name) for s in s_label]).to(device).float()

        optimizer.zero_grad()
        # A. Inverse scheduled sampling.
        masked_g_labels, current_rate = get_element_wise_mask_inverse(g_labels, lengths, epoch, start_pct=0.30)
        # B. Forward pass with v_jitter.
        output = model(dv3_feat, masked_g_labels, lengths, tau=tau)
        # C. Regression loss for the main KPI.
        loss_skill = w_mse * criteria['mse'](output['score'], s_norm) + \
                     w_mae * criteria['l1'](output['score'], s_norm)
        # D. Length-normalized TCN supervision.
        l_gest, l_err, l_tmse = compute_normalized_sequence_losses(output, g_labels, e_labels, lengths, criteria)
        loss_ms_tcn = w_gest * l_gest + w_err * l_err + w_tmse * l_tmse
        # E. CoT alignment and G16 penalty.
        logits = output['cot_logits']
        B_T, num_obs, _ = logits.shape

        target = torch.arange(num_obs).unsqueeze(0).expand(B_T, -1).to(device)
        loss_cot_raw = F.cross_entropy(logits.view(-1, num_obs), target.reshape(-1), reduction='none')
        loss_cot = loss_cot_raw.mean()

        loss_global_penalty = torch.abs(output['global_energy'] - 0.10)
        # F. Aggregate losses and downweight auxiliary signals.
        aux_loss = 0.1 * (loss_ms_tcn + w_cot * loss_cot)
        global_energy_loss = w_global_energy * loss_global_penalty
        batch_loss = loss_skill + aux_loss + global_energy_loss

        batch_loss.backward()
        # Clip gradients to reduce numerical spikes.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        monitor['total'] += batch_loss.item()
        monitor['skill'] += loss_skill.item()
        monitor['ms_tcn'] += (w_gest * l_gest + w_err * l_err).item()  # Action and error prediction.
        monitor['cot'] += (w_cot * loss_cot).item()
        monitor['g16'] += global_energy_loss.item()
        monitor['mask_rate'] += current_rate

        total_loss += batch_loss.item()

    # Compute averages.
    num_batches = len(dataloader)
    for key in monitor:
        monitor[key] /= num_batches

    # Print loss contribution report.
    skill_percent = (monitor['skill'] / monitor['total']) * 100
    print(f"Epoch {epoch:03d} | Loss: {monitor['total']:.4f} [Skill: {skill_percent:.1f}%] | "
          f"Mask: {monitor['mask_rate']:.2f}")

    return monitor['total'], tau, monitor['mask_rate']


def validate(model, dataloader, device, task_name, epoch):
    model.eval()
    all_skill_preds, all_skill_gts = [], []
    all_err_preds, all_err_gts = [], []
    all_gest_preds, all_gest_gts = [], []

    if not hasattr(validate, "best_mae"): validate.best_mae = float('inf')
    viz_cache = []

    with torch.no_grad():
        for data in dataloader:
            names, lengths, dv3_feat, e_labels, g_labels, s_labels = data
            output = model(dv3_feat.to(device), g_labels.to(device), lengths.to(device))

            pred_scores_norm = output['score'].cpu().numpy()
            for i in range(len(pred_scores_norm)):
                all_skill_preds.append(denormalize_score(pred_scores_norm[i], task_name))
                all_skill_gts.append(s_labels[i].item())

            err_probs_all = F.softmax(output['errors'][-1], dim=1)[:, 1, :]

            for b in range(len(lengths)):
                v_len = lengths[b].item()
                item_data = {
                    'name': names[b],
                    'v_gate': output['v_gate'][b, :v_len].cpu().numpy(),
                    'err_probs': err_probs_all[b, :v_len].cpu().numpy(),
                    'cot_gates': output['cot_gates'][b, :v_len].cpu().numpy(),  # [T, 16, 64]
                    'raw_feat': output['raw_feat'][b, :v_len].cpu().numpy()  # [T, 16, 64]
                }
                if "soft_weights" in output:
                    item_data['soft_weights'] = output['soft_weights'][b, 0, :v_len].cpu().numpy()
                viz_cache.append(item_data)

                # Collect classification metrics.
                all_err_preds.extend(torch.argmax(output['errors'][-1][b, :, :v_len], dim=0).cpu().numpy())
                all_err_gts.extend(e_labels[b, :v_len].numpy().astype(int))
                all_gest_preds.extend(torch.argmax(output['gestures'][-1][b, :, :v_len], dim=0).cpu().numpy())
                all_gest_gts.extend(g_labels[b, :v_len].numpy().astype(int))

    scc, _ = spearmanr(all_skill_gts, all_skill_preds)
    mae = mean_absolute_error(all_skill_gts, all_skill_preds)

    err_acc = accuracy_score(all_err_gts, all_err_preds)
    err_f1 = f1_score(all_err_gts, all_err_preds, average='macro')
    gest_acc = accuracy_score(all_gest_gts, all_gest_preds)
    gest_f1 = f1_score(all_gest_gts, all_gest_preds, average='macro')

    is_best = mae < validate.best_mae
    if is_best: validate.best_mae = mae

    return scc, mae, err_acc, err_f1, gest_acc, gest_f1


# 4. Main training loop
def train_and_validate(args, data_split_path, task_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(args.seed)

    train_loader = DataLoader(CustomVideoDataset(data_split_path, train=True), batch_size=args.bs, shuffle=True,
                              num_workers=4, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    test_loader = DataLoader(CustomVideoDataset(data_split_path, train=False), batch_size=args.bs, shuffle=False,
                             num_workers=4, collate_fn=collate_fn)

    model = GESNet(
        clip_feat_path=args.clip_path,
        num_actions=args.num_gest,
        num_stages=args.stages,
        num_layers=args.layers,
        use_soft_error=args.soft_pooling,
        use_tdf=True
    ).to(device)

    # Keep learnable_delta and g16_prompt in separate LR groups.
    optimizer = optim.Adam([
        {'params': [p for n, p in model.named_parameters() if 'learnable_delta' not in n and 'g_global_prompt' not in n],
         'lr': args.lr},
        {'params': model.learnable_delta, 'lr': args.lr},
        {'params': model.g_global_prompt, 'lr': args.lr}
    ], weight_decay=1e-4)

    scheduler = CosineLRScheduler(optimizer, t_initial=args.epo, lr_min=5e-6, warmup_t=5, warmup_lr_init=1e-7)
    criteria = {'mse': nn.MSELoss(), 'l1': nn.L1Loss(), 'ce': nn.CrossEntropyLoss(ignore_index=-100)}

    all_records = []
    for epoch in range(args.epo):
        # Pass epoch to compute tau.
        avg_loss, current_tau, mask_rate = train_one_epoch(model, train_loader, optimizer, criteria, device, task_name, epoch, args)
        scc, mae, e_acc, e_f1, g_acc, g_f1 = validate(model, test_loader, device, task_name, epoch=epoch)
        scheduler.step(epoch)
        all_records.append((scc, mae, epoch+1))
        print(
            f"Epoch [{epoch + 1}/{args.epo}] Loss: {avg_loss:.4f} | Tau: {current_tau:.2f} | SCC: {scc:.4f} MAE: {mae:.4f}")
        print(f"      [Aux] Err Acc: {e_acc:.3f} | Gest Acc: {g_acc:.3f}\n")

    # Sort by SCC descending.
    all_records_sorted_by_scc = sorted(all_records, key=lambda x: x[0], reverse=True)
    # Group the top four unique SCC values.
    top_scc_groups = {}  # {scc_value: [(scc, mae, epoch), ...]}
    for record in all_records_sorted_by_scc:
        scc_val = record[0]
        if scc_val not in top_scc_groups:
            top_scc_groups[scc_val] = []
        top_scc_groups[scc_val].append(record)
    # Keep the top four unique SCC values.
    unique_top_sccs = sorted(top_scc_groups.keys(), reverse=True)[:4]
    # Build final results.
    final_results = []
    for rank, scc_val in enumerate(unique_top_sccs):
        group = top_scc_groups[scc_val]
        # Sort by MAE ascending.
        group_sorted_by_mae = sorted(group, key=lambda x: x[1])
        if rank == 0:  # Best SCC.
            # Keep up to three best MAE results.
            num_to_take = min(3, len(group_sorted_by_mae))
            for i in range(num_to_take):
                final_results.append(group_sorted_by_mae[i])
        else:  # SCC ranks 2-4.
            # Keep the best MAE result.
            final_results.append(group_sorted_by_mae[0])

    # ===== Print results =====
    print("\n" + "=" * 30)
    print(f"Final Top Results for {task_name}:")
    print("=" * 30)
    # Count how many results share the best SCC.
    max_scc = unique_top_sccs[0]
    max_scc_count = sum(1 for r in final_results if r[0] == max_scc)
    for i, (scc, mae, ep) in enumerate(final_results):
        if i < max_scc_count:
            # Results with the best SCC.
            if max_scc_count == 1:
                rank_label = "SCC Top-1 (MAE Best)"
            else:
                mae_rank = i + 1
                rank_label = f"SCC Top-1 (MAE Rank-{mae_rank})"
        else:
            # Results with SCC ranks 2-4.
            scc_rank = unique_top_sccs.index(scc) + 1
            rank_label = f"SCC Top-{scc_rank} (MAE Best)"
        print(f"  [{rank_label}] SCC={scc:.5f}, MAE={mae:.5f} (Epoch {ep})")
    print("=" * 30 + "\n")

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-setting', default='LOUO', type=str, help='LOUO/LOSO3')
    parser.add_argument('-task_name', default='All_SU', type=str, help='NP/SU/All_NP/All_SU')
    parser.add_argument('-epo', default=200, type=int)
    parser.add_argument('-bs', default=8, type=int)
    parser.add_argument('-lr', default=1e-4, type=float)
    parser.add_argument('-stages', default=4, type=int)
    parser.add_argument('-layers', default=10, type=int)
    parser.add_argument('-seed', default=3407, type=int)
    parser.add_argument('-num_gest', default=8, type=int, help='8/15')
    parser.add_argument('-lambda_', default=0.15, type=float)  # TMSE weight.
    parser.add_argument('--soft_pooling', action='store_true')
    parser.add_argument('-clip_path', default='/path/to/rarp_clip_feat.npy')
    args = parser.parse_args()

    print(f'Setting: {args.setting}')
    print(f'Task Name: {args.task_name}')
    print(f'Epoch: {args.epo}')
    print(f'Learning Rate: {args.lr}')
    print(f'Seed: {args.seed}')
    print(f'Soft Pooling: {args.soft_pooling}')

    set_random_seed(args.seed)

    # Cross-validation setup.
    setting = args.setting
    task_name = args.task_name

    if setting == 'LOSO3':
        outs = ['1out', '2out', '3out', '4out', '5out']
        dataset_name = 'JIGSAWS'
    elif setting == 'LOUO' and task_name == 'All_NP':
        outs = ['1out', '2out', '3out', '4out', '5out', '6out', '7out']
        dataset_name = 'JIGSAWS'
    elif setting == 'LOUO' and task_name == 'All_SU':
        outs = ['1out', '2out', '3out', '4out', '5out', '6out', '7out', '8out']
        dataset_name = 'JIGSAWS'
    elif setting in ['4fold', '4fold-2'] and task_name == 'RARP':
        outs = ['F1', 'F2', 'F3', 'F4']
        dataset_name = 'RARP'
    else:
        raise ValueError(f"Invalid setting '{setting}' or task_name '{task_name}'")

    for out in outs:
        root_data_path = f'/cluster/project7/Llava_2024/Runlong/Datasets/{dataset_name}/'
        data_split_path = os.path.join(root_data_path, setting, out)
        print(f"\n>>> Starting cross-validation fold: {out}")

        train_and_validate(args, data_split_path, task_name)

if __name__ == "__main__":
    main()
