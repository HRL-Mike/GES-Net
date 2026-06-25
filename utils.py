import numpy as np


def get_element_wise_mask_inverse(g_labels, lengths, epoch, start_pct=0.40):
    """
    g_labels: [B, T]
    lengths: [B]
    epoch: current training epoch
    start_pct: initial mask ratio over total frames
    """
    masked_labels = g_labels.clone()
    B, T = g_labels.shape
    # Inverse mask schedule.
    # Decrease by 0.5% each epoch until it reaches zero.
    mask_rate = max(start_pct - (epoch * 0.005), 0.0)

    for i in range(B):
        L = lengths[i].item()
        if L < 10: continue  # Skip very short sequences.
        # 1. Use the middle 80% as the sampling range.
        start = int(L * 0.1)
        end = int(L * 0.9)
        mid_indices = np.arange(start, end)
        mid_len = len(mid_indices)

        if mid_len <= 0: continue
        # 2. Compute the mask count from total length L.
        num_to_mask = int(L * mask_rate)
        # 3. Clamp the count to the sampled range length.
        num_to_mask = min(num_to_mask, mid_len)

        if num_to_mask > 0:
            # Randomly mask indices in the middle range.
            masked_indices = np.random.choice(mid_indices, num_to_mask, replace=False)
            masked_labels[i, masked_indices] = -100

    return masked_labels, mask_rate
