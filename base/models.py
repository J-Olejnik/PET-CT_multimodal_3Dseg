import torch
from torch import nn
from monai.networks.nets import SwinUNETR
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv3d(x)
        x = self.batch(x)
        x = self.relu(x)
        return x
    
class scSEBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
    
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.sSE = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        cse_out = self.cSE(x) * x
        sse_out = self.sSE(x) * x
        return cse_out + sse_out
    
class ConvLSTM2DCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )
        self.hidden_dim = hidden_dim

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM3D(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTM2DCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        h = torch.zeros(B, self.cell.hidden_dim, H, W, device=x.device)
        c = torch.zeros_like(h)

        outputs = []

        for d in range(D):
            # [B, C, H, W]
            x_d = x[:, :, d, :, :]

            # Apply ConvLSTM cell on 2D slices
            h, c = self.cell(x_d, h, c)

            # [B, H, 1, H, W]
            outputs.append(h.unsqueeze(2))

        # [B, H, D, H, W]
        output = torch.cat(outputs, dim=2)
        return output
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, use_addons=False, lstm_hidden_size=None):
        super().__init__()
        self.use_addons = use_addons
        conv_blocks = []

        for i in range(num_convs):
            conv_blocks.append(ConvBlock(in_channels if i == 0 else out_channels, out_channels))
        
        self.conv = nn.Sequential(*conv_blocks)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        if use_addons:
            self.lstm = ConvLSTM3D(out_channels, lstm_hidden_size or out_channels)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)

        if self.use_addons:
            x = self.lstm(x)

        return x, p
        
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, use_addons=False, lstm_hidden_size=None):
        super().__init__()
        self.use_addons = use_addons
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        conv_blocks = []

        for i in range(num_convs):
            conv_blocks.append(ConvBlock(out_channels * 2 if i == 0 else out_channels, out_channels))
        
        self.conv = nn.Sequential(*conv_blocks)

        # Optional additional skipscSE and LSTM blocks
        if use_addons:
            self.skip_scse = scSEBlock(out_channels)
            self.lstm = ConvLSTM3D(out_channels, lstm_hidden_size or out_channels)
    
    def forward(self, x, skip):
        # Upsample
        x = self.upconv(x)
    
        # scSE attention
        if self.use_addons:
            skip = self.skip_scse(skip)

        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
            
        x = self.conv(x)

        # convLSTM
        if self.use_addons:
            x = self.lstm(x)
        
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, use_addons=False, lstm_hidden_size=None):
        super().__init__()
        self.use_addons = use_addons
        
        # Encoder blocks
        self.encoder1 = EncoderBlock(in_channels, 64, 2).to('cuda:0')
        self.encoder2 = EncoderBlock(64, 128, 2).to('cuda:0')
        self.encoder3 = EncoderBlock(128, 256, 3).to('cuda:0')
        
        self.bottleneck = EncoderBlock(256, 512, 3, use_addons=use_addons, lstm_hidden_size=lstm_hidden_size).to('cuda:0')
        
        # Decoder blocks with optional skipscSE and LSTM blocks
        self.decoder3 = DecoderBlock(512, 256, 3, use_addons=use_addons, lstm_hidden_size=lstm_hidden_size).to('cuda:0')
        self.decoder2 = DecoderBlock(256, 128, 2, use_addons=use_addons, lstm_hidden_size=lstm_hidden_size).to('cuda:0')
        self.decoder1 = DecoderBlock(128, 64, 2, use_addons=use_addons, lstm_hidden_size=lstm_hidden_size).to('cuda:1')
        
        # Output layer
        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1).to('cuda:1')
    
    def forward(self, x):
        x = x.to('cuda:0')
        # Encoder
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        
        # Bottleneck
        bottleneck, _ = self.bottleneck(p3)

        # s2 = s2.to('cuda:1')
        s1 = s1.to('cuda:1')
        
        # Decoder with skip connections
        d3 = self.decoder3(bottleneck, s3)
        # d3 = d3.to('cuda:1')
        d2 = self.decoder2(d3, s2)
        d2 = d2.to('cuda:1')
        d1 = self.decoder1(d2, s1)
        
        # Output
        output = self.final_conv(d1)
        
        return output

class SwinUNetR(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, feature_size: int = 48, dropout_rate: float = 0.2):
        super().__init__()
        self.model = SwinUNETR(
            img_size=(160, 128, 160),
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=feature_size,
            drop_rate = dropout_rate,
            use_checkpoint=True,
        )

    def forward(self, x):
        return self.model(x)
    
    