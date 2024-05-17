import torch
import torch.nn as nn

from modules import DWBlock_RGB, DWBlock_Freq, SRMConv2d_simple, SelectiveFusion, SeparableConv2d, TEFBlock, MSRMConv

class LDSFNet(nn.Module):

    def __init__(self, in_chans: int = 3, out_chans: int = 256, num_classes: int = 2):
        super(LDSFNet, self).__init__()
        # ---------- RGB stem ---------- #
        self.dw1_rgb = DWBlock_RGB(in_chans, 32, kernel_size=3, stride=1, padding=1)  # [b, 32, H, W]
        self.dw2_rgb = DWBlock_RGB(32, 64, kernel_size=3, stride=2, padding=1)  # [b, 64, H/2, W/2]
        self.dw3_rgb = DWBlock_RGB(64, 128, kernel_size=3, stride=2, padding=1)  # [b, 128, H/4, W/4]
        self.dw4_rgb = DWBlock_RGB(128, out_chans, kernel_size=3, stride=2, padding=1)  # [b, 256, H/8, W/8]
        self.dw5_rgb = nn.Sequential(SeparableConv2d(out_chans, out_chans, kernel_size=3, stride=2, padding=1),
                                     nn.BatchNorm2d(out_chans),
                                     nn.Hardswish(inplace=True))  # [b, 256, H/16, W/16]

        # cdc enhance
        self.cdc_enhance1_3 = TEFBlock(in_channels=in_chans, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.cdc_enhance3_5 = TEFBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

        # ---------- Freq stem ---------- #
        # learnable SRM
        self.lsrm = SRMConv2d_simple(learnable=True)
        # backbone
        self.dw1_fre = DWBlock_Freq(in_chans, 32, kernel_size=3, stride=1, padding=1)  # [b, 32, H/2, W/2]
        self.dw2_fre = DWBlock_Freq(32, 64, kernel_size=3, stride=2, padding=1)  # [b, 64, H/4, W/4]
        self.dw3_fre = DWBlock_Freq(64, 128, kernel_size=3, stride=2, padding=1)  # [b, 128, H/8, W/8]
        self.dw4_fre = DWBlock_Freq(128, out_chans, kernel_size=3, stride=2, padding=1)  # [b, 256, H/16, W/16]
        self.dw5_fre = nn.Sequential(SeparableConv2d(out_chans, out_chans, kernel_size=3, stride=2, padding=1),
                                     nn.BatchNorm2d(out_chans),
                                     nn.Hardswish(inplace=True))  # [b, 256, H/16, W/16]

        # multi-scale SRM
        self.msrm1 = MSRMConv(inc=32, outc=32)
        self.msrm3 = MSRMConv(inc=128, outc=128)

        # ---------- Selective Fusion ---------- #
        self.fusion = SelectiveFusion(channels=out_chans)

        # ---------- FC ---------- #
        self.fc = nn.Linear(out_chans, num_classes)

        self._init_weights()

    def _init_weights(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get batch
        batch_num = x.size(0)
        # ---------- RGB stem ---------- #
        x_rgb = self.dw1_rgb(x)
        # get cdc enhance1_3
        x_enhance = self.cdc_enhance1_3(x)  # [b, 64, H/2, W/2]
        # get multi-scale SRM1
        srm1 = self.msrm1(x_rgb)  # [b, 32, H, W]

        x_rgb = self.dw2_rgb(x_rgb)  # [b, 64, H/2, W/2]
        x_rgb = self.dw3_rgb(x_rgb + x_enhance)  # [b, 128, H/4, W/4]

        # get cdc enhance3_5
        x_enhance = self.cdc_enhance3_5(x_rgb)  # [b, 256, H/8, W/8]
        # get multi-scale SRM3
        srm3 = self.msrm3(x_rgb)  # [b, 128, H/4, W/4]

        x_rgb = self.dw4_rgb(x_rgb)
        x_rgb = self.dw5_rgb(x_rgb + x_enhance)

        # ---------- Freq stem ---------- #
        # get high-fre
        x_fre = self.lsrm(x)

        x_fre = self.dw1_fre(x_fre)
        x_fre = self.dw2_fre(x_fre + srm1)  # [b, 64, H/2, W/2]
        x_fre = self.dw3_fre(x_fre)  # [b, 128, H/4, W/4]
        x_fre = self.dw4_fre(x_fre + srm3)
        x_fre = self.dw5_fre(x_fre)

        # ---------- Selective Fusion ---------- #
        x_fus = self.fusion(x_rgb, x_fre)  # [b, 256, 1, 1]
        x_fus = self.fc(x_fus.reshape(batch_num, -1))

        return x_fus  # #

if __name__ == "__main__":
    a = torch.rand((1, 3, 256, 256))
    m = LDSFNet()
    m.eval()
    b = m(a)
    print(b.shape)
    pass
