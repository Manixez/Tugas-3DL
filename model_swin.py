import torch
import torch.nn as nn
import timm

class SwinTiny(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(SwinTiny, self).__init__()
        # Load pretrained Swin Transformer Tiny model
        self.model = timm.create_model('swin_tiny_patch4_window7_224', 
                                     pretrained=pretrained,
                                     num_classes=num_classes)
        
        # Note: timm 1.0+ uses ClassifierHead yang sudah include global pooling
        # Jadi kita tidak perlu mengubah head lagi, timm sudah handle dengan benar
        # via parameter num_classes di create_model

    def forward(self, x):
        # Forward pass melalui model Swin Transformer
        return self.model(x)

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test model dengan 5 kelas (gado_gado, rendang, bakso, soto_ayam, nasi_goreng)
    model = SwinTiny(num_classes=5).to(device)
    # Print model summary
    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224).to(device)  # Swin Tiny expects 224x224 images
    print(model(x).shape)  # Should print torch.Size([1, 5])