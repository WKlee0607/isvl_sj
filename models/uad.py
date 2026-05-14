import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LocalContextBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, drop=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        branch_specs = [(kernel_size, 1), (5, 1), (3, 2)]
        self.dwconvs = nn.ModuleList([
            nn.Conv2d(
                dim,
                dim,
                kernel_size=k,
                padding=(k // 2) * dilation,
                dilation=dilation,
                groups=dim,
            )
            for k, dilation in branch_specs
        ])
        self.branch_weights = nn.Parameter(torch.zeros(len(branch_specs)))
        self.pwconv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.pwconv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, side):
        B, N, C = x.shape
        if N != side * side:
            return x

        shortcut = x
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, side, side)
        branch_weights = torch.softmax(self.branch_weights, dim=0)
        branch_outputs = torch.stack([conv(x) for conv in self.dwconvs], dim=0)
        x = (branch_weights.view(-1, 1, 1, 1, 1) * branch_outputs).sum(dim=0)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pwconv2(x)
        x = x.flatten(2).transpose(1, 2)
        return shortcut + torch.tanh(self.gate) * self.drop(x)


class GraphContextBlock(nn.Module):
    def __init__(self, dim, drop=0., use_diagonal=True):
        super().__init__()
        self.use_diagonal = use_diagonal
        self.norm = nn.LayerNorm(dim)
        self.self_proj = nn.Linear(dim, dim)
        self.neighbor_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, side):
        B, N, C = x.shape
        if N != side * side:
            return x

        shortcut = x
        x = self.norm(x)
        x_grid = x.reshape(B, side, side, C)

        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.use_diagonal:
            offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        neighbor_sum = torch.zeros_like(x_grid)
        neighbor_count = torch.zeros(1, side, side, 1, device=x.device, dtype=x.dtype)
        for dy, dx in offsets:
            src_y0 = max(0, -dy)
            src_y1 = side - max(0, dy)
            dst_y0 = max(0, dy)
            dst_y1 = side - max(0, -dy)
            src_x0 = max(0, -dx)
            src_x1 = side - max(0, dx)
            dst_x0 = max(0, dx)
            dst_x1 = side - max(0, -dx)

            neighbor_sum[:, dst_y0:dst_y1, dst_x0:dst_x1] += x_grid[:, src_y0:src_y1, src_x0:src_x1]
            neighbor_count[:, dst_y0:dst_y1, dst_x0:dst_x1] += 1

        neighbor_mean = neighbor_sum / neighbor_count.clamp_min(1)
        neighbor_mean = neighbor_mean.reshape(B, N, C)

        message = self.self_proj(x) + self.neighbor_proj(neighbor_mean)
        message = self.out_proj(self.act(message))
        return shortcut + torch.tanh(self.gate) * self.drop(message)


class GraphAttentionContextBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            drop=0.,
            attn_drop=0.,
            use_diagonal=True,
            include_self=True,
            neighbor_radii=(1, 2),
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_diagonal = use_diagonal
        self.include_self = include_self
        self.neighbor_radii = neighbor_radii

        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.drop = nn.Dropout(drop)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def _offsets(self):
        offsets = []
        if self.include_self:
            offsets.append((0, 0))

        for radius in self.neighbor_radii:
            offsets += [(-radius, 0), (radius, 0), (0, -radius), (0, radius)]
            if self.use_diagonal:
                offsets += [
                    (-radius, -radius),
                    (-radius, radius),
                    (radius, -radius),
                    (radius, radius),
                ]
        return offsets

    def _shift_with_mask(self, x_grid, dy, dx):
        B, side, _, C = x_grid.shape
        shifted = torch.zeros_like(x_grid)
        mask = torch.zeros(1, side, side, 1, device=x_grid.device, dtype=torch.bool)

        src_y0 = max(0, -dy)
        src_y1 = side - max(0, dy)
        dst_y0 = max(0, dy)
        dst_y1 = side - max(0, -dy)
        src_x0 = max(0, -dx)
        src_x1 = side - max(0, dx)
        dst_x0 = max(0, dx)
        dst_x1 = side - max(0, -dx)

        shifted[:, dst_y0:dst_y1, dst_x0:dst_x1] = x_grid[:, src_y0:src_y1, src_x0:src_x1]
        mask[:, dst_y0:dst_y1, dst_x0:dst_x1] = True
        return shifted.reshape(B, side * side, C), mask.reshape(1, side * side, 1)

    def forward(self, x, side):
        B, N, C = x.shape
        if N != side * side:
            return x

        shortcut = x
        x = self.norm(x)
        x_grid = x.reshape(B, side, side, C)

        neighbors = []
        masks = []
        for dy, dx in self._offsets():
            shifted, mask = self._shift_with_mask(x_grid, dy, dx)
            neighbors.append(shifted)
            masks.append(mask)

        neighbor_count = len(neighbors)
        neighbors = torch.stack(neighbors, dim=2)
        valid_mask = torch.stack(masks, dim=2)

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(neighbors).reshape(B, N, neighbor_count, self.num_heads, self.head_dim)
        v = self.v_proj(neighbors).reshape(B, N, neighbor_count, self.num_heads, self.head_dim)

        attn = (q.unsqueeze(2) * k).sum(dim=-1) * self.scale
        attn = attn.masked_fill(~valid_mask, torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        message = (attn.unsqueeze(-1) * v).sum(dim=2).reshape(B, N, C)
        message = self.out_proj(message)
        return shortcut + torch.tanh(self.gate) * self.drop(message)


class INP_Former(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            aggregation,
            decoder,
            target_layers =[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],
            prototype_token=None,
            # embed_dim = None
    ) -> None:
        super(INP_Former, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.aggregation = aggregation
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.prototype_token = prototype_token[0]
        dim = self.prototype_token.shape[-1]
        self.local_context = nn.ModuleList([
            LocalContextBlock(dim=dim, kernel_size=3, drop=0.)
            for _ in range(len(self.decoder))
        ])
        self.patch_fusion_weights = nn.Parameter(torch.zeros(len(self.target_layers)))
        # self.encoder_fusion_weights = nn.ParameterList([
        #     nn.Parameter(torch.zeros(len(layer_idxs)))
        #     for layer_idxs in self.fuse_layer_encoder
        # ])
        self.decoder_fusion_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(len(layer_idxs)))
            for layer_idxs in self.fuse_layer_decoder
        ])
        self.graph_context = GraphAttentionContextBlock(
            dim=dim,
            num_heads=8,
            drop=0.,
            attn_drop=0.,
            use_diagonal=True,
            include_self=True,
            neighbor_radii=(1, 2),
        )

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

        # 单通道预测头（用于两个decoder输出）
        # self.pred_head_1 = nn.Conv2d(embed_dim, 1, kernel_size=1)
        # self.pred_head_2 = nn.Conv2d(embed_dim, 1, kernel_size=1)

        # # 可学习融合权重，初始化为0.5
        # self.alpha = nn.Parameter(torch.tensor(0.5))


    def gather_loss(self, query, keys, keep_ratio=0.8):
        self.distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        self.distance, self.cluster_index = torch.min(self.distribution, dim=2)

        keep_num = max(1, int(self.distance.shape[1] * keep_ratio))
        threshold = torch.kthvalue(self.distance.detach(), keep_num, dim=1, keepdim=True).values
        confident_mask = self.distance <= threshold
        gather_loss = (self.distance * confident_mask.float()).sum() / confident_mask.float().sum().clamp_min(1.0)
        return gather_loss

    def prototype_diversity_loss(self, prototypes=None):
        if prototypes is None:
            prototypes = self.prototype_token
        if prototypes.dim() == 3:
            prototypes = prototypes.mean(dim=0)

        prototypes = F.normalize(prototypes, dim=-1)
        sim = prototypes @ prototypes.t()
        eye = torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
        return ((sim - eye) ** 2).mean()

    def forward(self, x, use_gather_loss=True, return_patch_tokens=False):
        B = x.shape[0]
        en_list = self.extract_encoder_features(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        # Fused encoder patch tokens
        patch_tokens = self.fuse_feature(en_list, self.patch_fusion_weights)
        # x = self.fuse_feature(en_list)

        agg_prototype = self.prototype_token
        for i, blk in enumerate(self.aggregation):
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), patch_tokens)
        if use_gather_loss:
            g_loss = self.gather_loss(patch_tokens, agg_prototype)
        else:
            g_loss = torch.tensor(0.0, device=patch_tokens.device)

        x = patch_tokens
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)
        x = self.graph_context(x, side) # GAT

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, agg_prototype)
            x = self.local_context[i](x, side) # LocalContextBlock
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        # en = [
        #     self.fuse_feature(
        #         [en_list[idx] for idx in idxs],
        #         self.encoder_fusion_weights[group_idx]
        #     )
        #     for group_idx, idxs in enumerate(self.fuse_layer_encoder)
        # ]
        de = [
            self.fuse_feature(
                [de_list[idx] for idx in idxs],
                self.decoder_fusion_weights[group_idx] # learnable fusion
            )
            for group_idx, idxs in enumerate(self.fuse_layer_decoder)
        ]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        # anomaly map from two decoder outputs
        # anomaly_map_1 = self.pred_head_1(de[0])  # shape: [B, 1, H, W]
        # anomaly_map_2 = self.pred_head_2(de[1])  # shape: [B, 1, H, W]

        # # 加权融合
        # anomaly_map = self.alpha * anomaly_map_1 + (1 - self.alpha) * anomaly_map_2
        # anomaly_map = torch.sigmoid(anomaly_map)  # 映射为概率图

        if return_patch_tokens:
            return en, de, g_loss, patch_tokens, agg_prototype
        return en, de, g_loss, agg_prototype

    def extract_encoder_features(self, x):
        if getattr(self.encoder, "is_dinov3", False):
            with torch.no_grad():
                outputs = self.encoder.get_intermediate_layers(
                    x,
                    n=self.target_layers,
                    return_class_token=True,
                    return_extra_tokens=True,
                    norm=False,
                )
            en_list = []
            for patch_tokens, class_token, extra_tokens in outputs:
                class_token = class_token.unsqueeze(1)
                en_list.append(torch.cat([class_token, extra_tokens, patch_tokens], dim=1))
            return en_list

        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        return en_list

    def fuse_feature(self, feat_list, weights=None):
        feats = torch.stack(feat_list, dim=1)
        if weights is None:
            return feats.mean(dim=1)

        weights = torch.softmax(weights, dim=0).view(1, -1, 1, 1)
        return (feats * weights).sum(dim=1)









































