import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class CrossChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CrossChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, xS, xT):
        avg_out = self.fc(self.avg_pool(xT))
        max_out = self.fc(self.max_pool(xT))
        out = avg_out + max_out
        return xS * self.sigmoid(out)


class ClassificationHead(nn.Module):
    def __init__(self,
                 num_labels:int=7, 
                 dim_in: int = 2048,
                 dropout_rate:float = 0.5,
                 is_3d:bool = True) -> None:
        super(ClassificationHead, self).__init__()
        self.is_3d = is_3d
        if self.is_3d:
            self.output_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout=nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.proj=nn.Linear(dim_in, num_labels, bias=True)

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)
        # # Performs projection.
        if self.is_3d:
            x = x.permute((0, 2, 3, 4, 1)) # BTHWC
            x = self.proj(x)
            x = x.permute((0, 4, 1, 2, 3)) # BCTHW
        else:
            x = self.proj(x)

        if self.is_3d:
            # Performs global averaging.
            x = self.output_pool(x)
            x = x.view(x.shape[0], -1)
        return x


class ClassificationHeadWithAttention(ClassificationHead):
    def __init__(self,
                 num_labels:int=3, 
                 dim_in: int = 2048,
                 dropout_rate:float = 0.5,
                 is_3d:bool = True) -> None:
        super(ClassificationHeadWithAttention, self).__init__(num_labels=num_labels, 
                                                        dim_in = dim_in,
                                                        dropout_rate = dropout_rate,
                                                        is_3d = is_3d)
        self.ch_attn = ChannelAttention(channel=dim_in)

    def forward(self, x):
        x = self.ch_attn(x)
        if self.dropout:
            x = self.dropout(x)
        # # Performs projection.
        if self.is_3d:
            x = x.permute((0, 2, 3, 4, 1))
            x = self.proj(x)
            x = x.permute((0, 4, 1, 2, 3))
        else:
            x = self.proj(x)

        if self.is_3d is not None:
            # Performs global averaging.
            x = self.output_pool(x)
            x = x.view(x.shape[0], -1)
        return x
   
 
class ClassificationHeadWithCrossAttention(ClassificationHead):
    def __init__(self,
                 num_labels:int=3, 
                 dim_in: int = 2048,
                 dropout_rate:float = 0.5,
                 is_3d:bool = True) -> None:
        super(ClassificationHeadWithCrossAttention, self).__init__(num_labels=num_labels, 
                                                        dim_in = dim_in,
                                                        dropout_rate = dropout_rate,
                                                        is_3d = is_3d)
        self.ch_attn = CrossChannelAttention(channel=dim_in)

    def forward(self, xS, xT):
        x = self.ch_attn(xS, xT)
        if self.dropout:
            x = self.dropout(x)
        # # Performs projection.
        if self.is_3d:
            x = x.permute((0, 2, 3, 4, 1))
            x = self.proj(x)
            x = x.permute((0, 4, 1, 2, 3))
        else:
            x = self.proj(x)

        if self.is_3d is not None:
            # Performs global averaging.
            x = self.output_pool(x)
            x = x.view(x.shape[0], -1)
        return x


class MultiLabelHead(nn.Module):
    def __init__(self, 
                 num_labels:int=7, 
                 in_feats: int = 2048, 
                 dropout_rate:float = 0.5,
                 is_3d:bool = True):
        super(MultiLabelHead, self).__init__()
        self.cls_head = ClassificationHead(num_labels=num_labels, 
                                           dim_in = in_feats,
                                           dropout_rate = dropout_rate,
                                           is_3d = is_3d)
    def forward(self, x):
        return self.cls_head(x)


class MultiClassSingleHead(nn.Module):
    def __init__(self, 
                 num_labels:int=9, 
                 in_feats: int = 2048, 
                 dropout_rate:float = 0.5,
                 is_3d:bool = True):
        super(MultiClassSingleHead, self).__init__()
        assert num_labels==9, f"num_labels should be 9"
        self.cls_head = ClassificationHead(num_labels=num_labels, 
                                           dim_in = in_feats,
                                           dropout_rate = dropout_rate,
                                           is_3d = is_3d)

    def forward(self, x):
        x = self.cls_head(x)
        return x[:,:3], x[:,3:6], x[:,6:9]
    

class MultiClassMultiHead(nn.Module):
    def __init__(self, 
                 num_classes:int=3,
                 in_feats: int = 2048, 
                 dropout_rate:float = 0.5,
                 is_3d:bool = True):
        super(MultiClassMultiHead, self).__init__()
        assert num_classes==3, f"num_labels should be 3"
        self.rain_head = ClassificationHead(num_labels=num_classes, 
                                           dim_in = in_feats,
                                           dropout_rate = dropout_rate,
                                           is_3d = is_3d)
        
        self.fog_head = ClassificationHead(num_labels=num_classes, 
                                           dim_in = in_feats,
                                           dropout_rate = dropout_rate,
                                           is_3d = is_3d)

        self.snow_head = ClassificationHead(num_labels=num_classes, 
                                           dim_in = in_feats,
                                           dropout_rate = dropout_rate,
                                           is_3d = is_3d)

    def forward(self, x):
        rx = self.rain_head(x)
        fx = self.fog_head(x)
        sx = self.snow_head(x)
        return rx, fx, sx


class MultiClassMultiHeadAttention(nn.Module):
    def __init__(self, 
                 num_classes:int=3,
                 in_feats: int = 2048, 
                 dropout_rate:float = 0.5,
                 is_3d:bool = True):
        super(MultiClassMultiHeadAttention, self).__init__()
        assert num_classes==3, f"num_labels should be 3"
        self.rain_head = ClassificationHeadWithAttention(num_labels=num_classes, 
                                                        dim_in = in_feats,
                                                        dropout_rate = dropout_rate,
                                                        is_3d = is_3d)
        self.fog_head = ClassificationHeadWithAttention(num_labels=num_classes, 
                                                        dim_in = in_feats,
                                                        dropout_rate = dropout_rate,
                                                        is_3d = is_3d)
        self.snow_head = ClassificationHeadWithAttention(num_labels=num_classes, 
                                                        dim_in = in_feats,
                                                        dropout_rate = dropout_rate,
                                                        is_3d = is_3d)

    def forward(self, x):
        rx = self.rain_head(x)
        fx = self.fog_head(x)
        sx = self.snow_head(x)
        return rx, fx, sx
    
    
class MMConvClassificationHead(nn.Module):
    def __init__(self, 
                 head: str = 'mlh',
                 num_classes:int=3,
                 in_feats: int = 2048, 
                 dropout_rate:float = 0.5,
                 is_3d:bool = True):
        super(MMConvClassificationHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_feats*2, 512, 1, stride=1, bias=True),
            nn.ReLU(),
            )
        self.cls = create_head(name=head, num_labels=num_classes, in_feats=512, dropout_rate=dropout_rate, is_3d=is_3d)
        
    def forward(self, xS, xT):
        x = torch.cat([torch.stack([xS[:,i], xT[:,i]], dim=1) for i in range(xS.shape[1])], dim=1).contiguous()
        x = self.conv(x)
        return self.cls(x)


class MMCrossAttentionClassificationHead(nn.Module):
    def __init__(self, 
                 num_classes:int=3,
                 in_feats: int = 2048, 
                 dropout_rate:float = 0.5,
                 is_3d:bool = True):
        super(MMCrossAttentionClassificationHead, self).__init__()
        assert num_classes==3, f"num_labels should be 3"
        self.rain_head = ClassificationHeadWithCrossAttention(num_labels=num_classes, 
                                                              dim_in = in_feats,
                                                              dropout_rate = dropout_rate,
                                                              is_3d = is_3d)
        self.fog_head = ClassificationHeadWithCrossAttention(num_labels=num_classes, 
                                                             dim_in = in_feats,
                                                             dropout_rate = dropout_rate,
                                                             is_3d = is_3d)
        self.snow_head = ClassificationHeadWithCrossAttention(num_labels=num_classes, 
                                                              dim_in = in_feats,
                                                              dropout_rate = dropout_rate,
                                                              is_3d = is_3d)

    def forward(self, xS, xT):
        rx = self.rain_head(xS, xT)
        fx = self.fog_head(xS, xT)
        sx = self.snow_head(xS, xT)
        return rx, fx, sx


__name2head__ = {'mlh': MultiLabelHead, 
                 'mcsh': MultiClassSingleHead, 
                 'mcmh': MultiClassMultiHead, 
                 'mcmha': MultiClassMultiHeadAttention}


def create_head(name:str='mlh', num_labels:int=7, in_feats:int=2048, dropout_rate:float=0.5, is_3d:bool=True):
    if name in list(__name2head__.keys()):
        head = __name2head__[name](num_labels, in_feats, dropout_rate, is_3d)
    else:
        raise NotImplementedError(f"{name} is not implemented.")
    return head


def create_opt_head(consensus:str='conv3d', name:str='mlh', num_labels:int=7, in_feats:int=2048, dropout_rate:float=0.5, is_3d:bool=True):
    if consensus=='conv3d':
        head = MMConvClassificationHead(head=name,
                                        num_classes=num_labels,
                                        in_feats=in_feats, 
                                        dropout_rate=dropout_rate,
                                        is_3d=is_3d)
    elif consensus=='crossattn':
        head = MMCrossAttentionClassificationHead(num_classes = num_labels,
                                                  in_feats = in_feats, 
                                                  dropout_rate = dropout_rate,
                                                  is_3d = is_3d)
    else:
        raise NotImplementedError(f"{name} is not implemented.")
    
    return head
