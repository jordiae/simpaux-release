import torch.nn as nn
import torch.nn.functional as F
import torch


@torch.jit.script
def silu(x):
    # TODO: https://twitter.com/apaszke/status/1188584949374476295
    return x * torch.sigmoid(x)


class FCLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=None, dropout=0.0,
                 **kwargs):
        assert activation in (None, 'relu', 'selu', 'swish-1')
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc = nn.Linear(in_features, out_features, **kwargs)

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'selu':
            self.act = nn.SELU()
        elif activation == 'swish-1':
            self.act = silu
        elif activation is None:
            self.act = nn.Identity()
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation, batchnorm=True,
                 dropout=0.0, kernel_size=3, affine=True, conv_bias=False, bn_epsilon=1e-05, bn_momentum=0.1, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.affine = affine

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                padding=kernel_size//2, bias=conv_bias, **kwargs)
        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels, affine=affine, eps=bn_epsilon, momentum=bn_momentum)
        else:
            self.bn = torch.nn.Identity()
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'selu':
            self.act = nn.SELU()
        elif activation == 'swish-1':
            self.act = silu

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x, betas=None, gammas=None):
        x = self.conv2d(x)
        x = self.bn(x)
        # FIXME: cond BN
        if betas is not None and gammas is not None and not self.affine:
            gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
            betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
            x = x * gammas + betas

        x = self.act(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 num_layers, max_pool_kernel_size=None,
                 has_max_pool=False, activation='swish-1', batchnorm=True,
                 dropout=0.0, kernel_size=3, conditioning=False, conv_bias=False, bn_epsilon=1e-05, bn_momentum=0.1,
                 **conv2d_kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_pool_kernel_size = max_pool_kernel_size
        self.num_layers = num_layers
        self.conditioning = conditioning

        self.conv2d_layers = []

        self.shortcut = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1, stride=1)
        for i in range(num_layers):
            # if conditioning is used, turn off BN's affine transform for the
            # last conv layer in the block (will be overridden in forward)
            affine = not conditioning or i < num_layers - 1
            conv2d_kwargs['affine'] = affine
            conv2d_kwargs['conv_bias'] = conv_bias
            conv2d_kwargs['bn_epsilon'] = bn_epsilon
            conv2d_kwargs['bn_momentum'] = bn_momentum
            self.conv2d_layers.append(
                Conv2dLayer(in_channels if i == 0 else out_channels,
                            out_channels, activation, batchnorm, dropout,
                            kernel_size, **conv2d_kwargs)
            )
            self.add_module(f'Conv2dLayer_{i:02d}', self.conv2d_layers[-1])

        try:
            if isinstance(self.conv2d_layers[-1].bn, nn.BatchNorm2d):
                setattr(self.conv2d_layers[-1].bn, 'zero_init', True)
        except BaseException as e:
            print(e)
        if has_max_pool and max_pool_kernel_size is not None:
            self.pool = nn.MaxPool2d(max_pool_kernel_size)

    def forward(self, x, betas=None, gammas=None):
        x_shortcut = self.shortcut(x)
        for i, layer in enumerate(self.conv2d_layers):
            if i == self.num_layers - 1 and self.conditioning:
                x = layer(x, betas=betas, gammas=gammas)
            else:
                x = layer(x)
        x = x + x_shortcut

        if hasattr(self, 'pool'):
            x = self.pool(x)
        return x


class PrototypicalNetwork(nn.Module):
    def __init__(self, in_channels,
                 num_layers_per_block, num_blocks, max_pool_kernel_size=None,
                 num_channels=64, num_channels_growth_factor=1,
                 num_max_pools=0, activation='relu', batchnorm=True,
                 dropout=0.0, kernel_size=3, conv_bias=False, bn_epsilon=1e-05, bn_momentum=0.1,
                 conditioning=False):
        assert activation in ['relu', 'selu', 'swish-1']
        super().__init__()

        self.in_channels = in_channels

        num_channels = [num_channels * num_channels_growth_factor**i
                        for i in range(num_blocks)]

        self.num_channels = num_channels
        self.conditioning = conditioning

        self.blocks = []

        for i in range(num_blocks):
            has_max_pool = i < num_max_pools
            self.blocks.append(
                ResidualBlock(
                    in_channels=in_channels if i == 0 else num_channels[i-1],
                    out_channels=num_channels[i],
                    num_layers=num_layers_per_block,
                    max_pool_kernel_size=max_pool_kernel_size,
                    has_max_pool=has_max_pool,
                    activation=activation,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    kernel_size=kernel_size,
                    conditioning=conditioning,
                    conv_bias=conv_bias,
                    bn_epsilon=bn_epsilon,
                    bn_momentum=bn_momentum
                )
            )
            self.add_module(f'ResBlock_{i:02d}', self.blocks[-1])
        print()

    def forward(self, inputs, betas=None, gammas=None):
        x = inputs.view(-1, *inputs.shape[2:])
        if self.conditioning:
            betas = torch.split(betas, self.num_channels, dim=1)
            gammas = torch.split(gammas, self.num_channels, dim=1)
        else:
            betas = [None] * len(self.blocks)
            gammas = [None] * len(self.blocks)
        for b, block in enumerate(self.blocks):
            x = block(x, betas=betas[b], gammas=gammas[b])

        # average pooling over spatial axes
        embeddings = F.avg_pool2d(x, kernel_size=x.shape[-2:])

        return embeddings.view(*inputs.shape[:2], -1)


class Bridge(nn.Module):
    def __init__(self, in_features, num_hid_features, out_features, num_hid_layers,
                 activation='relu', dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_hid_layers = num_hid_layers
        self.num_hid_features = num_hid_features
        self.activation = activation

        self.layers = nn.ModuleList()
        for i in range(num_hid_layers):
            self.layers.append(FCLayer(
                in_features if i == 0 else num_hid_features,
                num_hid_features, activation=activation, dropout=dropout))
        self.layers.append(FCLayer(
            num_hid_features if num_hid_layers > 0 else in_features,
            out_features * 2,
            activation=None
        ))

    def forward(self, x):
        for l, layer in enumerate(self.layers):
            x = layer(x)
        betas, gammas = torch.split(x, self.out_features, dim=1)
        return betas, gammas + 1.0  # network outputs offsets from gamma=1


class AuxiliarNetwork(nn.Module):
    def __init__(self, in_channels,
                 num_layers_per_block, num_blocks, n_outputs, max_pool_kernel_size=None,
                 num_channels=64, num_channels_growth_factor=1,
                 num_max_pools=0, activation='relu', batchnorm=True, dropout=0.0, kernel_size=3, n_extra_linear=1,
                 protonet=False,
                 conv_bias=False, bn_epsilon=1e-05, bn_momentum=0.1):
          
        assert activation in ['relu', 'selu', 'swish-1']
        super().__init__()

        self.in_channels = in_channels

        self.protonet = protonet

        num_channels = [num_channels * num_channels_growth_factor**i
                        for i in range(num_blocks)]

        self.num_channels = num_channels

        self.blocks = []

        for i in range(num_blocks):
            has_max_pool = i < num_max_pools
            self.blocks.append(
                ResidualBlock(
                    in_channels=in_channels if i == 0 else num_channels[i-1],
                    out_channels=num_channels[i],
                    num_layers=num_layers_per_block,
                    max_pool_kernel_size=max_pool_kernel_size,
                    has_max_pool=has_max_pool,
                    activation=activation,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    kernel_size=kernel_size,
                    conv_bias=conv_bias,
                    bn_epsilon=bn_epsilon,
                    bn_momentum=bn_momentum
                )
            )
            self.add_module(f'ResBlock_{i:02d}', self.blocks[-1])

        if not protonet:
            if n_extra_linear > 0:
                layers = []
                for i in range(n_extra_linear):
                    layers_ = [nn.Linear(num_channels[-1], num_channels[-1])]
                    if batchnorm:
                        layers_.append(nn.BatchNorm1d(num_channels[-1]))
                    layers_.append(nn.ReLU())
                    if dropout > 0:
                        layers_.append(nn.Dropout(dropout))
                    layers.extend(layers_)
                self.linear = nn.Sequential(*layers)
            self.n_extra_linear = n_extra_linear

            self.classifier = nn.Linear(num_channels[-1], n_outputs)

    def forward(self, inputs, return_aux_outputs=True, return_features=False,
                feature_layers=[]):

        if inputs.ndim == 5:
            x = inputs.view(-1, *inputs.shape[2:])
        else:
            x = inputs
        feat_rvals = []
        for b, block in enumerate(self.blocks):
            x = block(x)
            if b in feature_layers and return_features:
                feat_rvals.append(x)

        if self.protonet:
            rvals = []
            if self.return_features:
                rvals.append(feat_rvals)
            if self.return_aux_outputs:
                # average pooling over spatial axes
                embeddings = F.avg_pool2d(x, kernel_size=x.shape[-2:])
                rvals.append(embeddings.view(*inputs.shape[:2], -1))
            return rvals if len(rvals) > 1 else rvals[0]

        x = F.avg_pool2d(x, kernel_size=x.shape[-2:]).flatten(1)

        # # average pooling over spatial axes
        # embeddings = F.avg_pool2d(x, kernel_size=x.shape[-2:])

        # if inputs.ndim == 5:
        #     # FIXME: aren't these two doing exactly the opposite of each other?
        #     features = embeddings.view(*inputs.shape[:2], -1)
        #     features = features.view(features.shape[0] * features.shape[1], features.shape[2])
        # else:
        #     features = embeddings.view(inputs.shape[0], -1)

        if self.n_extra_linear > 0:
            x = self.linear(x)

        # if none of the res block feature maps is requested, we append the
        # last fc layer's output
        rvals = []
        if return_features:
            if len(feat_rvals) == 0:
                feat_rvals.append(x)
            rvals.append(feat_rvals)
        if return_aux_outputs:
            rvals.append(self.classifier(x))
        return rvals if len(rvals) > 1 else rvals[0]


# TODO: should we really set default args here?
class SimpAux(nn.Module):
    def __init__(self,
                 # PrototypicalNetwork args
                 in_channels,
                 num_layers_per_block, num_blocks, aux_num_layers_per_block,
                 aux_num_blocks,
                 max_pool_kernel_size=None,
                 num_channels=64, num_channels_growth_factor=1,
                 num_max_pools=0,
                 dropout=0.0, kernel_size=3,
                 # Bridge-specific args
                 bridge_num_hid_features=256, bridge_num_hid_layers=1,
                 bridge_dropout=0.0,
                 # AuxiliarNetwork-specific args
                 aux_n_outputs=1,
                 aux_n_extra_linear=1,
                 aux_dropout=0.0,
                 aux_num_channels=32,
                 # shared args
                 activation='relu',
                 aux_backprop=False,
                 bridge_input_aux_layers=[],

                 conv_bias=False,
                 bn_epsilon=1e-05,
                 bn_momentum=0.1
                 ):
        super().__init__()
        self.bridge_input_aux_layers = bridge_input_aux_layers

        self.protonet = PrototypicalNetwork(
            in_channels=in_channels,
            num_layers_per_block=num_layers_per_block,
            num_blocks=num_blocks,
            max_pool_kernel_size=max_pool_kernel_size,
            num_channels=num_channels,
            num_channels_growth_factor=num_channels_growth_factor,
            num_max_pools=num_max_pools,
            dropout=dropout,
            kernel_size=kernel_size,
            activation=activation,
            batchnorm=True,
            conditioning=True,
            conv_bias=conv_bias,
            bn_epsilon=bn_epsilon,
            bn_momentum=bn_momentum
        )
        # TODO: aux net could possibly be shallower structure
        self.aux_net = AuxiliarNetwork(
            in_channels=in_channels,
            n_outputs=aux_n_outputs,
            num_layers_per_block=aux_num_layers_per_block,
            num_blocks=aux_num_blocks,
            max_pool_kernel_size=max_pool_kernel_size,
            num_channels=aux_num_channels,
            num_channels_growth_factor=num_channels_growth_factor,
            num_max_pools=num_max_pools,
            dropout=aux_dropout,
            kernel_size=kernel_size,
            activation=activation,
            batchnorm=True,
            n_extra_linear=aux_n_extra_linear,
            conv_bias=conv_bias,
            bn_epsilon=bn_epsilon,
            bn_momentum=bn_momentum
        )

        if len(bridge_input_aux_layers):
            self.bridge_num_in_features = sum(self.aux_net.num_channels[i]
                                              for i in bridge_input_aux_layers)
        else:
            self.bridge_num_in_features = self.aux_net.num_channels[-1]

        self.bridge = Bridge(
            in_features=self.bridge_num_in_features,
            num_hid_features=bridge_num_hid_features,
            out_features=sum(self.protonet.num_channels),
            num_hid_layers=bridge_num_hid_layers,
            activation=activation,
            dropout=bridge_dropout)
        self.aux_backprop = aux_backprop

    def parameters_wo_aux(self):
        return list(self.protonet.parameters()) + list(self.bridge.parameters())

    def named_parameters_wo_aux(self):
        named_parameters = dict(self.protonet.named_parameters())
        named_parameters.update(self.bridge.named_parameters())
        return named_parameters.items()

    def forward(self, x, summary_writer=None, step=None,
                return_aux_outputs=False):
        assert not ((summary_writer is None) ^ (step is None))

        aux_rval = self.aux_net(x,
                                return_aux_outputs=return_aux_outputs,
                                return_features=True,
                                feature_layers=self.bridge_input_aux_layers)
        if return_aux_outputs:
            aux_feat, aux_out = aux_rval
        else:
            aux_feat = aux_rval

        # detach aux features to not backprop classfication loss into aux net
        if not self.aux_backprop:
            aux_feat = [af.detach() for af in aux_feat]
        # pool 2d feature maps to vectorize them
        aux_feat = [
            F.avg_pool2d(af, kernel_size=af.shape[-2:]).flatten(1)
            if af.ndim == 4 else af
            for af in aux_feat]
        aux_feat = torch.cat(aux_feat, dim=-1)

        # FIXME: pool and concat aux_feat elements
        # if input is 2d feature map, use avg pooling to turn it into a vector
        # if x.ndim == 4:
        #     x = F.avg_pool2d(x, kernel_size=x.shape[-2:]).flatten(1)

        betas, gammas = self.bridge(aux_feat)

        if summary_writer is not None:
            start = 0
            for block_idx, block_n_channels in \
                    enumerate(self.protonet.num_channels):
                summary_writer.add_histogram(
                    f'debug/betas_{block_idx}_hist',
                    betas[:, start:start + block_n_channels],
                    global_step=step)
                summary_writer.add_histogram(
                    f'debug/gammas_{block_idx}_hist',
                    gammas[:, start:start + block_n_channels],
                    global_step=step)

                summary_writer.add_image(
                    f'debug/betas_{block_idx}',
                    betas[:, start:start + block_n_channels],
                    dataformats='HW',
                    global_step=step)
                summary_writer.add_image(
                    f'debug/gammas_{block_idx}',
                    gammas[:, start:start + block_n_channels],
                    dataformats='HW',
                    global_step=step)
                start += block_n_channels
            summary_writer.add_histogram(
                f'debug/aux_feat_hist',
                aux_feat,
                global_step=step)
            summary_writer.add_image(
                'debug/aux_feat',
                aux_feat,
                dataformats='HW',
                global_step=step)

        embeddings = self.protonet(x, betas=betas, gammas=gammas)

        if return_aux_outputs:
            return embeddings, aux_out
        return embeddings


if __name__ == '__main__':
    resblock = ResidualBlock(in_channels=3, out_channels=64,
                             num_layers=3, max_pool_kernel_size=2)
    x = torch.randn(16, 3, 84, 84)
    y = resblock(x)

    protonet = PrototypicalNetwork(in_channels=3,
                                   num_layers_per_block=3, num_blocks=3,
                                   max_pool_kernel_size=2)
    y = protonet(x)
    import ipdb
    ipdb.set_trace()
