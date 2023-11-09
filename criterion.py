from itertools import permutations
import torch

EPS = 1e-8

def cal_loss(source, estimate_source, source_lengths,PIT=False):
    """
    Args:
        source: [B, C, T]
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    if PIT:
        max_snr, perms, max_snr_idx = cal_si_snr_with_pit(source,
                                                        estimate_source,
                                                        source_lengths)
        loss = 0 - torch.mean(max_snr)
        reorder_estimate_source = reorder_source(estimate_source, perms, max_snr_idx)
        return loss, max_snr, estimate_source, reorder_estimate_source
    else:
        si_snr = cal_si_snr(source,estimate_source, source_lengths)
        loss = 0 - torch.mean(si_snr)


def cal_si_snr(source, estimate_source, source_lengths):
    assert source.size() == estimate_source.size()
    B, C, K, L = source.size()
    num_samples = (L* source_lengths).view(-1, 1, 1, 1).float()  # [B, 1, 1, 1]
    mean_target = torch.sum(source, dim=[2, 3], keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=[2, 3], keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    mask = get_mask(source, source_lengths)
    zero_mean_target *= mask
    zero_mean_estimate *= mask
    s_target = zero_mean_target.view(B, C, -1)  # [B, C, T]
    s_estimate = zero_mean_estimate.view(B, C, -1)  # [B, C, T]
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=2, keepdim=True)  # [B, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=2, keepdim=True) + EPS  # [B, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, T]

    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, T]

    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C]
    si_snr = torch.mean(pair_wise_si_snr, dim=-1, keepdim = True)

    return si_snr


def cal_si_snr_with_pit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, K, L]
        estimate_source: [B, C, K, L]
        source_lengths: [B], each item is between [0, K]
    """
    assert source.size() == estimate_source.size()
    B, C, K, L = source.size()
    
    num_samples = (L* source_lengths).view(-1, 1, 1, 1).float()  # [B, 1, 1, 1]
    mean_target = torch.sum(source, dim=[2, 3], keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=[2, 3], keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along K
    mask = get_mask(source, source_lengths)
    zero_mean_target *= mask
    zero_mean_estimate *= mask


    # flat K, L to T (T = K * L)
    flat_target = zero_mean_target.view(B, C, -1)  # [B, C, T]
    flat_estimate = zero_mean_estimate.view(B, C, -1)  # [B, C, T]
    s_target = torch.unsqueeze(flat_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(flat_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    #这部分相当于两两相乘
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)

    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C
    return max_snr, perms, max_snr_idx

def cal_snr_with_pit(source, estimate_source, source_lengths):
    # [B,C,T]
    B,C,T = source.size()
    mean_target = torch.sum(source, dim=2, keepdim=True) / T
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / T
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    mask = get_mask(source, source_lengths)
    zero_mean_estimate *= mask
    zero_mean_target *= mask 
    
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    e_noise = s_target - s_estimate # [B, C, C, T]
    s_target = s_target.repeat(1,C,1,1)

    pair_snr = 10 * torch.log10(torch.sum(s_target**2,dim=-1)/(torch.sum(e_noise**2,dim=-1) + EPS) + EPS)
    # print(pair_snr.shape)

    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    snr_set = torch.einsum('bij,pij->bp', [pair_snr, perms_one_hot])

    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C #为啥要除以C？

    return max_snr , perms, max_snr_idx
    

def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, K, L]
    """
    B, C, *_ = source.size()
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)

    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source


def get_mask(source, source_lengths):
    B,C,T= source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        # print(source_lengths[i])
        mask[i, :, source_lengths[i]:] = 0
    return mask

if __name__ == '__main__':
    a = torch.randn([5,2,3])
    b = torch.randn([5,2,3])
    # l = torch.randint([5])
    l = [1,2,2,1,1]

    print(cal_snr_with_pit(a,b,l))
