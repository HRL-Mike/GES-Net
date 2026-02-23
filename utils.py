import numpy as np


def get_element_wise_mask_inverse(g_labels, lengths, epoch, start_pct=0.35):
    masked_labels = g_labels.clone()
    B, T = g_labels.shape
    mask_rate = max(start_pct - (epoch * 0.005), 0.0)
    for i in range(B):
        L = lengths[i].item()
        if L < 10: continue
        start = int(L * 0.1)
        end = int(L * 0.9)
        mid_indices = np.arange(start, end)
        mid_len = len(mid_indices)

        if mid_len <= 0: continue
        num_to_mask = int(L * mask_rate)
        num_to_mask = min(num_to_mask, mid_len)

        if num_to_mask > 0:
            masked_indices = np.random.choice(mid_indices, num_to_mask, replace=False)
            masked_labels[i, masked_indices] = -100

    return masked_labels, mask_rate
