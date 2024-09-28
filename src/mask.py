import math
from multiprocessing import Value
from tabulate import tabulate
import torch
import numpy as np

## inspired from https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py


class MaskCollator(object):

    def __init__(
        self,
        allow_overlap: bool,
        min_context_share: float,
        max_context_share: float,
        min_target_share: float,
        max_target_share: float,
        num_preds: int,
        num_encs: int,
        num_features: int,
        cardinalities: list,
    ):
        super(MaskCollator, self).__init__()

        assert min_context_share < max_context_share, "Min < Max !"
        assert min_target_share < max_target_share, "Min < Max !"

        self.allow_overlap = allow_overlap

        self.min_context_share = min_context_share
        self.max_context_share = max_context_share

        self.max_target_share = max_target_share
        self.min_target_share = min_target_share

        self._itr_counter = Value("i", -1)
        self.num_preds = num_preds
        self.num_encs = num_encs

        self.num_features = num_features
        self.cardinalities = cardinalities

        err_msg = "Max and min shares are too close."

        self.min_context = round(num_features * self.min_context_share)
        self.max_context = round(num_features * self.max_context_share)
        assert self.max_context > self.min_context, err_msg
        assert self.min_context > 0, "Min context is 0."

        self.max_target = round(num_features * self.max_target_share)
        self.min_target = round(num_features * self.min_target_share)
        assert self.max_context > self.min_context, err_msg
        assert self.min_target > 0, "Min target is 0."

        self.print_params()

    def print_params(self):
        print(f"{self.__class__.__name__} params:")
        print(
            tabulate(
                [
                    ["Allow overlap", self.allow_overlap],
                    ["Min context share", self.min_context_share],
                    ["Max context share", self.max_context_share],
                    ["Min target share", self.min_target_share],
                    ["Max target share", self.max_target_share],
                    ["Min context", self.min_context],
                    ["Max context", self.max_context],
                    ["Min target", self.min_target],
                    ["Max target", self.max_target],
                    ["Num preds", self.num_preds],
                    ["Num encs", self.num_encs],
                    ["Num features", self.num_features],
                    ["Cardinalities", self.cardinalities],
                ]
            )
        )

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, batch):
        n_batch = len(batch)

        batch = [b[0] for b in batch]
        n_features = len(batch[0])

        seed = self.step()
        gen = torch.Generator()
        gen.manual_seed(seed)

        n_mskd_cxt_ftrs, n_mskd_trgt_ftrs = math.inf, math.inf
        while self.num_encs * n_mskd_cxt_ftrs + n_mskd_trgt_ftrs > self.num_features:
            n_mskd_cxt_ftrs = self._sample_num_mask(
                generator=gen, _min=self.min_context, _max=self.max_context
            )[0]
            n_mskd_trgt_ftrs = self._sample_num_mask(
                generator=gen, _min=self.min_target, _max=self.max_target
            )[0]

        mask_ctx = []
        mask_trgt = []
        for _ in range(n_batch):
            m_ctx, m_trgt = self.create_masks(
                int(n_mskd_cxt_ftrs),
                int(n_mskd_trgt_ftrs),
                n_features,
                self.num_encs,
                self.num_preds,
            )
            mask_ctx.append(m_ctx)
            mask_trgt.append(m_trgt)

        collated_masks_trgt = torch.utils.data.default_collate(mask_trgt)
        collated_masks_ctx = torch.utils.data.default_collate(mask_ctx)
        collated_batch = torch.utils.data.default_collate(batch)

        return collated_batch, collated_masks_ctx, collated_masks_trgt

    def create_masks(
        self,
        n_mskd_cxt_ftrs: int,
        n_mskd_trgt_ftrs: int,
        n_features: int,
        n_encs: int,
        n_preds: int,
    ):

        if n_encs * n_mskd_cxt_ftrs + n_mskd_trgt_ftrs > n_features:
            raise ValueError("Sum of the masks is greater than the number of features.")

        mask_cxt = []
        mask_trgt = []
        all_indices = np.arange(n_features)

        while len(mask_cxt) < n_encs:
            np.random.shuffle(all_indices)
            mask_cxt.append(all_indices[:n_mskd_cxt_ftrs])
            all_indices = np.setdiff1d(all_indices, mask_cxt[-1])

        while len(mask_trgt) < n_preds:
            np.random.shuffle(all_indices)
            mask_trgt.append(all_indices[:n_mskd_trgt_ftrs])

        return mask_cxt, mask_trgt

    def _sample_num_mask(
        self,
        generator,
        _min=None,
        _max=None,
    ):

        num_hidden_feature = math.inf
        while num_hidden_feature > self.num_features:
            _rand = torch.rand(1, generator=generator).numpy()
            num_hidden_feature = _min + np.round((_max - _min) * _rand)
        return num_hidden_feature

    def _sample_masked_batch(self, mask_size, generator, acceptable_region=None):

        mask = self.random_binary_vector(
            self.num_features, mask_size, generator, acceptable_region
        )
        complement_mask = 1 - mask

        return mask, complement_mask

    def random_binary_vector(
        self, num_features, _size, generator, acceptable_region=None
    ):
        """
        Generate a random binary vector of length D with _size number of 1s.

        Parameters:
            D (int): Length of the vector.
            _size (int): Number of 1s in the vector.

        Returns:
            torch.Tensor: Random binary vector.
        """
        if acceptable_region is None:
            acceptable_region = torch.ones(num_features)

        vector = torch.zeros(num_features)

        allowed_indices = (acceptable_region == 1).nonzero().squeeze()
        if len(allowed_indices.shape) == 0:
            allowed_indices = allowed_indices.unsqueeze(dim=0)

        selected_indices = torch.randperm(allowed_indices.size(0), generator=generator)[
            :_size
        ]

        selected_indices = allowed_indices[selected_indices]

        vector[selected_indices] = 1

        return vector
