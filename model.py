import torch
from torch import nn
import torch.nn.functional as F

class GRN(nn.Module):
    """ Global Response Normalization, proposed in ConvNeXt v2 paper """

    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        # x = (B, C, T)
        # Want to average first over length (T), then divide by average channel (i.e. average of C)
        # Divide the L2 norms by the average for each channel
        Gx = x.norm(p=2, dim=2, keepdim=True) # (B, C, T) --> (B, C, 1)
        Nx = Gx / Gx.mean(dim=1, keepdim=True).clamp(min=self.eps) # (B, C, 1) / (B, 1, 1) --> (B, C, 1)
        return self.gamma * (x * Nx) + self.beta + x

class DropPath(nn.Module):
    """ DropPath regularisation can be used if needed, as described here:
    https://arxiv.org/abs/1605.07648v4
    """
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def drop_path(self, x, keep_prob: float = 1.0, inplace: bool = False):
        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep_prob)
        mask.div_(keep_prob)
        if inplace:
            x.mul_(mask)
        else:
            x = x * mask
        return x

    def forward(self, x):
        if self.training and self.p > 0:
            x = self.drop_path(x, self.p, self.inplace)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

class ConvNeXtBlock(nn.Module):
    # A 1D ConvNeXt v2 block
    def __init__(self, dim, drop_path_prob=0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=7, groups=dim, padding=3)
        self.norm = nn.LayerNorm(dim)
        self.pwconv_1 = nn.Conv1d(dim, 4*dim, kernel_size=1, padding=0)
        self.act = nn.GELU()
        self.GRN = GRN(4*dim)
        self.pwconv_2 = nn.Conv1d(4*dim, dim, kernel_size=1, padding=0)
        self.droppath = DropPath(p=drop_path_prob)

    def forward(self, inputs):
        # Inputs has shape (B, C, T)
        x = self.dwconv(inputs)
        x = self.norm(x.permute(0,2,1))
        x = x.permute(0,2,1) # Layernorm expects channels last
        x = self.pwconv_1(x)
        x = self.act(x)
        x = self.GRN(x)
        x = self.pwconv_2(x)
        return inputs + self.droppath(x)

class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.down = nn.Conv1d(in_dim, out_dim, kernel_size=7, stride=2, padding=3)

    def forward(self, x):
        x = self.norm(x.permute(0, 2, 1)).permute(0,2,1)
        x = self.down(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, out_dim=3, depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], drop_path_prob=0.5, dropout=0.5):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.initial_conv = nn.Conv1d(1, dims[0], kernel_size=7, stride=2, padding=3)
        self.initial_norm = nn.LayerNorm(dims[0])
        self.layers = nn.ModuleList()
        for i, dd in enumerate(zip(depths, dims)):
            depth, dim = dd
            for d in range(depth):
                self.layers.append(ConvNeXtBlock(dim, drop_path_prob=drop_path_prob))
            if i+1 != len(dims):
                self.layers.append(DownSample(in_dim=dim, out_dim=dims[i+1]))
            else:
                self.layers.append(DownSample(in_dim=dim, out_dim=dims[i]))
        self.final_norm = nn.LayerNorm(1024)
        self.flatten = nn.Flatten()
        self.output = nn.Sequential(
            nn.Linear(1024, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim)
        )

    def forward(self, x, shapes=False):
        x = self.initial_conv(x)
        x = self.initial_norm(x.permute(0,2,1)).permute(0,2,1)
        if shapes:
            print(x.shape)
        for l in self.layers:
            x = l(x)
            if shapes:
                print(x.shape)
        x = self.final_norm(F.gelu(self.flatten(x)))#.permute(0,2,1)).permute(0,2,1) # global average pooling, (B, C, T) -> (B, C)
        #x = self.flatten(x)
        if shapes:
            print(x.shape)
        x = self.output(x)
        return x

#| code-fold: true
class GLU(nn.Module):
    def __init__(self, in_dim, out_dim, act=F.gelu, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=bias)
        self.linear2 = nn.Linear(in_dim, out_dim, bias=bias)
        self.act = act

    def forward(self, x):
        return self.act(self.linear1(x))*self.linear2(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, pos_length=129, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.qk = nn.Linear(dim, 2*dim)
        self.v = nn.Linear(dim, dim)
        self.mhsa_out = nn.Linear(dim, dim, bias=False)
        self.GLU = GLU(dim, (dim*3)//2, bias=False)
        self.linear_out = nn.Linear((dim*3)//2, dim, bias=False)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.pos = nn.Embedding(pos_length, embedding_dim=dim)
        self.dropout = nn.Dropout(dropout)

    def mhsa(self, x):
        B, T, C = x.shape
        q, k = self.qk(x + self.pos(torch.arange(x.shape[1], device=x.device))).chunk(2, dim=-1)
        v = self.v(x)
        q = q.reshape(B, self.heads, T, C//self.heads)
        k = k.reshape(B, self.heads, T, C//self.heads)
        v = v.reshape(B, self.heads, T, C//self.heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.reshape(B, T, C)
        x = self.mhsa_out(x)
        return x

    def ffwd(self, x):
        x = self.GLU(x)
        x = self.linear_out(self.dropout(x))
        return x

    def forward(self, x):
        x = x + self.mhsa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class ConvNeXtTransformer(nn.Module):
    def __init__(self, datadim=2048, depths=[2, 2, 6, 2], dims=[40, 80, 160, 320],
                 transformer_layers=6, attention_heads=2, drop_path_prob=0.1, dropout=0.1):
        super().__init__()
        self.depths = depths
        self.datadim = datadim
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.initial_conv = nn.Conv1d(1, dims[0], kernel_size=7, stride=2, padding=3)
        self.initial_norm = nn.LayerNorm(dims[0])
        self.conv_layers = nn.ModuleList()
        self.transformer_layers = nn.ModuleList()
        self.cls_token = nn.Embedding(1, dims[-1])
        for i, dd in enumerate(zip(depths, dims)):
            depth, dim = dd
            for d in range(depth):
                self.conv_layers.append(ConvNeXtBlock(dim, drop_path_prob=drop_path_prob))
            if i+1 != len(dims):
                self.conv_layers.append(DownSample(in_dim=dim, out_dim=dims[i+1]))
        for i in range(transformer_layers):
            self.transformer_layers.append(TransformerBlock(dims[-1], attention_heads, dropout=dropout))
        self.final_norm = nn.LayerNorm(dims[-1])
        self.output_pure_impure = nn.Sequential(
            GLU(dims[-1], 32),
            nn.Linear(32, 1)
            )
        self.patch_classifier = nn.Conv1d(dims[-1], 16, kernel_size=1)

    def forward(self, x, shapes=False):
        x = self.initial_conv(x)
        x = self.initial_norm(x.permute(0,2,1)).permute(0,2,1)
        if shapes:
            print(x.shape)
        for l in self.conv_layers:
            x = l(x)
            if shapes:
                print(x.shape)
        x = F.gelu(x)
        # Now concatenate the CLS token with the output of the convolutional layers
        cls = self.cls_token(torch.arange(1, device=x.device)).squeeze().expand(x.shape[0], -1).unsqueeze(-1)
        x = torch.cat([cls, x], dim=-1)
        x = x.permute(0, 2, 1)
        for l in self.transformer_layers:
            x = l(x)
            if shapes:
                print(x.shape)
        x = self.final_norm(x).permute(0,2,1)
        if shapes:
            print(x.shape)
        x_pure_impure = self.output_pure_impure(x[:,:,0])
        patch_pure_impure = self.patch_classifier(x[:,:,1:])
        B, T, C = patch_pure_impure.shape
        patch_pure_impure = patch_pure_impure.permute(0,2,1).reshape(B, T*C)
        if shapes:
            print(x_pure_impure.shape, patch_pure_impure.shape)
        return x_pure_impure, patch_pure_impure