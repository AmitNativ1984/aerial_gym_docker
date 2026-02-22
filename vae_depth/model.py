import torch
import torch.nn as nn


class DepthEncoder(nn.Module):
    """Convolutional encoder for depth images.

    Architecture: 4 blocks of strided conv layers with BatchNorm + ELU,
    followed by 1x1 conv channel reduction, flatten, and FC head (512-dim hidden).

    Input:  [B, 1, 180, 320]
    Output: [B, 2 * latent_dim]  (concatenated mu and logvar)
    """

    def __init__(self, latent_dim: int = 32, fc_channel_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.elu = nn.ELU()

        # Block 0: 180x320 -> 90x160 -> 45x80
        self.conv0 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        self.bn0 = nn.BatchNorm2d(32)
        self.conv0_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn0_1 = nn.BatchNorm2d(32)

        # Block 1: 45x80 -> 23x40
        self.conv1_0 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn1_0 = nn.BatchNorm2d(64)
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)

        # Block 2: 23x40 -> 12x20
        self.conv2_0 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2_0 = nn.BatchNorm2d(128)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)

        # Block 3: 12x20 -> 6x10
        self.conv3_0 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3_0 = nn.BatchNorm2d(256)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)

        # 1x1 conv to reduce channels: 256 -> fc_channel_dim
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(256, fc_channel_dim, kernel_size=1),
            nn.ELU(),
        )

        # FC: fc_channel_dim * 6 * 10 -> 512 -> 2*latent_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_channel_dim * 6 * 10, 512),
            nn.ELU(),
            nn.Linear(512, 2 * latent_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("linear"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 0: [B,1,180,320] -> [B,32,45,80]
        x = self.elu(self.bn0(self.conv0(x)))
        x = self.elu(self.bn0_1(self.conv0_1(x)))

        # Block 1: [B,32,45,80] -> [B,64,23,40]
        x = self.elu(self.bn1_0(self.conv1_0(x)))
        x = self.elu(self.bn1_1(self.conv1_1(x)))

        # Block 2: [B,64,23,40] -> [B,128,12,20]
        x = self.elu(self.bn2_0(self.conv2_0(x)))
        x = self.elu(self.bn2_1(self.conv2_1(x)))

        # Block 3: [B,128,12,20] -> [B,256,6,10]
        x = self.elu(self.bn3_0(self.conv3_0(x)))
        x = self.elu(self.bn3_1(self.conv3_1(x)))

        # 1x1 reduce + flatten + FC
        x = self.channel_reduce(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DepthDecoder(nn.Module):
    """Transposed-convolutional decoder for depth images.

    Input:  [B, latent_dim]
    Output: [B, 1, 180, 320]
    """

    def __init__(self, latent_dim: int = 32, fc_channel_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc_channel_dim = fc_channel_dim

        # FC: latent_dim -> 512 -> fc_channel_dim * 6 * 10
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ELU(),
            nn.Linear(512, fc_channel_dim * 6 * 10),
            nn.ELU(),
        )

        # 1x1 conv to expand channels: fc_channel_dim -> 256
        self.channel_expand = nn.Sequential(
            nn.Conv2d(fc_channel_dim, 256, kernel_size=1),
            nn.ELU(),
        )

        # Deconv layers: 6x10 -> 12x20 -> 23x40 -> 45x80 -> 90x160 -> 180x320
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=(1, 1)),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("linear"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), self.fc_channel_dim, 6, 10)
        x = self.channel_expand(x)
        x = self.deconvs(x)
        return x


class DepthVAE(nn.Module):
    """Variational Autoencoder for depth image compression.

    Encoder compresses 320x180 depth images to a latent_dim-dimensional latent space.
    Decoder reconstructs from latent vectors.

    Interfaces:
        forward(x) -> (x_recon, mu, logvar, z)  -- training
        encode(x)  -> mu                         -- deterministic inference (RL)
        decode(z)  -> x_recon                    -- visualization
    """

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = DepthEncoder(latent_dim=latent_dim)
        self.decoder = DepthDecoder(latent_dim=latent_dim)

    def forward(self, x: torch.Tensor):
        z_params = self.encoder(x)
        mu = z_params[:, : self.latent_dim]
        logvar = torch.clamp(z_params[:, self.latent_dim :], min=-10.0, max=10.0)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic encoding: returns mu only (no sampling)."""
        z_params = self.encoder(x)
        mu = z_params[:, : self.latent_dim]
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
