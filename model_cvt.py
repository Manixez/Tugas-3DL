import torch
import torch.nn as nn
from transformers import CvtForImageClassification, CvtConfig

class CvT(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(CvT, self).__init__()
        
        if pretrained:
            # Load pretrained CvT-13 model from Hugging Face
            # CvT-13 adalah Convolutional vision Transformer dengan 13 layers
            self.model = CvtForImageClassification.from_pretrained(
                'microsoft/cvt-13',
                num_labels=num_classes,
                ignore_mismatched_sizes=True  # Allow different number of classes
            )
        else:
            # Create model from scratch with custom config
            config = CvtConfig(num_labels=num_classes)
            self.model = CvtForImageClassification(config)
        
    def forward(self, x):
        # CvT expects pixel_values as input
        outputs = self.model(pixel_values=x)
        return outputs.logits  # Return logits for classification

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test model with 10 classes (CIFAR-10)
    print("Loading CvT-13 model...")
    model = CvT(num_classes=10, pretrained=True).to(device)
    
    # Print model summary
    print("Model created successfully!")
    print(f"Model architecture: CvT-13 (Convolutional vision Transformer)")
    
    # Test with sample input
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224).to(device)  # CvT-13 expects 224x224 images
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should print torch.Size([2, 10])
    print(f"Expected: [2, 10] ✓" if output.shape == torch.Size([2, 10]) else f"Expected: [2, 10] ✗")