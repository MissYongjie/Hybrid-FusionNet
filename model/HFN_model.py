import torch
from torch import nn
from einops import rearrange
#
class Transformerencoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, num_layers, num_heads, dropout_prob, distilled=False):
        super(Transformerencoder, self).__init__()
        self.Transformerencoderlayer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            nhead=num_heads,
            dim_feedforward=out_channels,
            dropout=dropout_prob
        )

        self.Transformerencoder = nn.TransformerEncoder(
            encoder_layer=self.Transformerencoderlayer,
            num_layers=num_layers
        )

    def forward(self, x):                                  # 64 1 103 7 7
        x = x.transpose(0, 1)
        x = self.Transformerencoder(x)
        x = x.transpose(0, 1)
        return x
#
#
import math
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        input: (B, N, C)
        B = Batch size, N = patch_size * patch_size, C = dimension hidden_features and out_features
        output: (B, N, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=16, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        input: (B, N, C)
        B = Batch size, N = patch_size * patch_size, C = dimension for attention
        output: (B, N, C)
        """
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class patch_embed(nn.Module):
    def __init__(self, in_feature_map_size=7, in_chans=3, embed_dim=128, n_groups=1):
        super().__init__()
        self.ifm_size = in_feature_map_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, groups=n_groups)
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        input: (B, in_chans, in_feature_map_size, in_feature_map_size)
        output: (B, (after_feature_map_size x after_feature_map_size-2), embed_dim = C)
        """
        x = self.proj(x)  # 1 256 7 7
        x = self.relu(self.batch_norm(x))

        x = x.flatten(2).transpose(1, 2)  # 1 49 256

        after_feature_map_size = self.ifm_size

        return x, after_feature_map_size


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MyTransformer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 n_groups=1, embed_dims=64, num_heads=8, mlp_ratios=1, depths=2, num_stages=6):
        super().__init__()

        new_bands = math.ceil(in_chans / n_groups) * n_groups
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - in_chans))

        self.transformerencoders = Transformerencoder(
            in_channels=928, out_channels=128,
            num_classes=9, num_layers=6, num_heads=8,
            dropout_prob=0.1, distilled=False
        )

        self.patch_embed = patch_embed(
            in_feature_map_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dims,
            n_groups=n_groups
        )

        for i in range(num_stages):
            self.block = Block(
                dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                drop=0.,
                attn_drop=0.)

        self.norm = nn.LayerNorm(embed_dims)
        self.head = nn.Linear(embed_dims, num_classes)  # 只有pvt时的Head

    def forward(self, x):
        # (bs, 1, n_bands, patch size (ps, of HSI), ps)
        x = self.pad(x).squeeze(dim=1)  # 1, 104, 7, 7
        x, s = self.patch_embed(x)  # s = feature map size after patch embedding x=(1 49 256) s=7
        x = self.block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


#*************************CNNTransformerencoder****************************#
class CNNTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, num_layers, num_heads, dropout_prob):
        super(CNNTransformer, self).__init__()

        self.conv1_1 = nn.Sequential(
            nn.Conv1d(in_channels*num_classes, out_channels, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, out_channels),
            nn.Dropout(),
            nn.Linear(out_channels, num_classes),
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv3d(1, num_classes, kernel_size=(3, 3, 3), padding="same", bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(num_classes)
        )

        self.conv3_2 = nn.Sequential(
            nn.Conv3d(1, num_classes, kernel_size=(5, 3, 3), padding="same", bias=False),
            nn.BatchNorm3d(num_classes),
            nn.ReLU()
        )

        self.conv3_3 = nn.Sequential(
            nn.Conv3d(1, num_classes, kernel_size=(7, 3, 3), padding="same", bias=False),
            nn.BatchNorm3d(num_classes),
            nn.ReLU()
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels*num_classes, out_channels, kernel_size=(3, 3), padding="same", bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding="same", bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

        self.conv2_3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding="same", bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.flatten = nn.Flatten()
        self.linear2d = nn.Linear(3136, out_channels)
        self.linearfc = nn.Linear(out_channels, out_channels)
        self.linear = nn.Linear(out_channels, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.transformer = MyTransformer(img_size=7, in_chans=in_channels*num_classes, num_classes=num_classes, n_groups=1, depths=3)

# # transformer & 3dcnn
    def forward(self, x):
        # (b=batch_size, n=1, c=n_bands, h=patch size (ps, of HSI), w=ps)
        x_3d = self.conv3_1(x)
        x_ts = rearrange(x_3d, 'b n c h w -> b 1 (n c) h w')
        x_2d = rearrange(x_3d, 'b n c h w -> b (n c) h w')
        x_ts = self.transformer(x_ts)

        x_2d = self.conv2_1(x_2d)
        x_2d = self.conv2_2(x_2d)
        x_2d = self.conv2_2(x_2d)

        x_2d = self.flatten(x_2d)
        # x_2d = self.dropout(x_2d)
        x_2d = self.linear2d(x_2d)
        # x_2d = self.dropout(x_2d)
        x_2d = self.linearfc(x_2d)
        x_2d = self.linear(x_2d)

        x_2d = self.softmax(x_2d)

        x = torch.add(x_2d, x_ts)
        return x

def cnnt(dataset, patch_size):
    model = None
    if dataset == 'sa':
        model = CNNTransformer(in_channels=204, out_channels=64, num_classes=16, num_layers=3, num_heads=8, dropout_prob=0.1)
    elif dataset == 'pu':
        model = CNNTransformer(in_channels=103, out_channels=64, num_classes=9, num_layers=3, num_heads=8, dropout_prob=0.5)
    return model

if __name__ == '__main__':
    t = torch.randn(size=(64, 1, 204, 7, 7))
    print("input shape:", t.shape)
    net = cnnt(dataset='sa', patch_size=7)
    print("output shape:", net(t).shape)