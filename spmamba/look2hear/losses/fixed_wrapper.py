###
# Fixed-order (non-PIT) loss wrapper.
# Use when target channels have a fixed, meaningful order so the model
# learns a fixed channel assignment. For HLS-CMDS: s1=heart, s2=lung, so at
# inference on a raw mixture output channel 0 is heart and channel 1 is lung
# with no permutation ambiguity.
###

import torch
from torch import nn


class FixedOrderLossWrapper(nn.Module):
    """Non-permutation sibling of ``PITLossWrapper``.

    Estimate ``i`` is always matched to target ``i`` (no permutation search).
    It reuses the same pairwise loss function as PIT (e.g. ``pairwise_neg_snr``)
    and simply takes the diagonal of the ``[B, n_src, n_src]`` pairwise matrix,
    so per-pair loss values are numerically identical to the PIT version --
    only the assignment differs.

    Accepts the same ``pit_from`` / ``threshold_byloss`` kwargs as
    ``PITLossWrapper`` so a config swaps over by changing only the class name.
    """

    def __init__(
        self, loss_func, pit_from="pw_mtx", perm_reduce=None, threshold_byloss=False
    ):
        super().__init__()
        self.loss_func = loss_func
        self.pit_from = pit_from
        self.perm_reduce = perm_reduce
        self.threshold_byloss = threshold_byloss

    def forward(self, ests, targets, return_ests=False, **kwargs):
        # Pairwise matrix [B, n_src, n_src]; entry [b, i, j] = loss(est_i, target_j)
        pw_loss = self.loss_func(ests, targets, **kwargs)
        assert pw_loss.ndim == 3, (
            "FixedOrderLossWrapper expects a pairwise loss returning "
            f"[batch, n_src, n_src], got shape {tuple(pw_loss.shape)}"
        )
        assert pw_loss.shape[0] == targets.shape[0], "loss needs same batch dim as input"

        # Fixed assignment -> diagonal (est_i vs target_i), mean over sources.
        diag = torch.diagonal(pw_loss, dim1=-2, dim2=-1)  # [B, n_src]
        if diag.is_complex():
            diag = diag.real
        per_sample = diag.mean(dim=-1)  # [B]

        if self.threshold_byloss:
            kept = per_sample[per_sample > -30]
            if kept.nelement() > 0:
                per_sample = kept

        mean_loss = per_sample.mean()
        if return_ests:
            return mean_loss, ests
        return mean_loss


if __name__ == "__main__":
    from matrix import pairwise_neg_snr
    from pit_wrapper import PITLossWrapper

    ests = torch.randn(4, 2, 16000)
    targets = torch.randn(4, 2, 16000)
    fixed = FixedOrderLossWrapper(pairwise_neg_snr, threshold_byloss=False)
    pit = PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx", threshold_byloss=False)
    print("fixed:", fixed(ests, targets).item())
    print("pit  :", pit(ests, targets).item(), "(<= fixed, since PIT minimises)")
