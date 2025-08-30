import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class CompleteVGG19FeatureExtractor(nn.Module):
    """
    VGG19 Feature Extractor that includes ALL convolutional layers
    No layers are skipped - you get features from all 16 conv layers
    """
    
    def __init__(self, pretrained=True):
        super(CompleteVGG19FeatureExtractor, self).__init__()
        
        # Load pretrained VGG19
        vgg = models.vgg19(pretrained=pretrained)
        self.features = vgg.features
        
        # Map ALL 16 convolutional layers (no skipping!)
        self.all_conv_layers = {
            # Block 1
            '0': 'conv1_1',   
            '2': 'conv1_2',
            
            # Block 2
            '5': 'conv2_1',   
            '7': 'conv2_2',
            
            # Block 3
            '10': 'conv3_1',  
            '12': 'conv3_2',  
            '14': 'conv3_3',  
            '16': 'conv3_4',
            
            # Block 4
            '19': 'conv4_1',  
            '21': 'conv4_2',  
            '23': 'conv4_3',  
            '25': 'conv4_4',
            
            # Block 5
            '28': 'conv5_1',  
            '30': 'conv5_2',  
            '32': 'conv5_3',  
            '34': 'conv5_4'
        }
        
        # Traditional style transfer layers (subset of all)
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.content_layers = ['conv4_2']
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x, layers=None):
        """
        Extract features from specified layers
        
        Args:
            x: Input tensor [batch_size, 3, H, W]
            layers: List of layer names to extract. If None, extracts from all conv layers
            
        Returns:
            Dictionary with layer names as keys and feature tensors as values
        """
        if layers is None:
            layers = list(self.all_conv_layers.values())  # All 16 conv layers
        
        features = {}
        current_features = x
        
        for idx, layer in enumerate(self.features):
            current_features = layer(current_features)
            
            # Check if this layer should be saved
            layer_name = self.all_conv_layers.get(str(idx))
            if layer_name and layer_name in layers:
                features[layer_name] = current_features
        
        return features
    
    def get_all_conv_features(self, x):
        """Get features from ALL 16 convolutional layers"""
        return self.forward(x, list(self.all_conv_layers.values()))
    
    def get_block_features(self, x, block_num):
        """
        Get features from a specific block
        
        Args:
            x: Input tensor
            block_num: Block number (1, 2, 3, 4, or 5)
        """
        block_layers = {
            1: ['conv1_1', 'conv1_2'],
            2: ['conv2_1', 'conv2_2'], 
            3: ['conv3_1', 'conv3_2', 'conv3_3', 'conv3_4'],
            4: ['conv4_1', 'conv4_2', 'conv4_3', 'conv4_4'],
            5: ['conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
        }
        
        if block_num not in block_layers:
            raise ValueError(f"Block number must be 1-5, got {block_num}")
        
        return self.forward(x, block_layers[block_num])
    
    def get_traditional_style_features(self, x):
        """Get features from traditional style transfer layers (subset)"""
        return self.forward(x, self.style_layers)
    
    def get_content_features(self, x):
        """Get features from content layers"""
        return self.forward(x, self.content_layers)
    
    def get_last_conv_features(self, x):
        """Get output from the very last convolutional layer"""
        return self.forward(x, ['conv5_4'])['conv5_4']

def compare_layer_selections():
    """
    Compare different layer selection strategies
    """
    print("COMPARING DIFFERENT LAYER SELECTION STRATEGIES")
    print("=" * 55)
    
    # Create sample input
    x = torch.randn(1, 3, 224, 224)
    
    # Initialize extractor
    extractor = CompleteVGG19FeatureExtractor(pretrained=True)
    extractor.eval()
    
    with torch.no_grad():
        # 1. All conv layers
        all_features = extractor.get_all_conv_features(x)
        print(f"\n1. ALL CONV LAYERS ({len(all_features)} layers):")
        for name, features in all_features.items():
            print(f"   {name}: {features.shape}")
        
        # 2. Traditional style transfer selection
        style_features = extractor.get_traditional_style_features(x)
        print(f"\n2. TRADITIONAL STYLE SELECTION ({len(style_features)} layers):")
        for name, features in style_features.items():
            print(f"   {name}: {features.shape}")
        
        # 3. Block-wise extraction
        print(f"\n3. BLOCK-WISE EXTRACTION:")
        for block in range(1, 6):
            block_features = extractor.get_block_features(x, block)
            print(f"   Block {block}: {list(block_features.keys())}")
        
        # 4. Content layer
        content_features = extractor.get_content_features(x)
        print(f"\n4. CONTENT LAYER:")
        for name, features in content_features.items():
            print(f"   {name}: {features.shape}")
        
        # 5. Last conv layer
        last_features = extractor.get_last_conv_features(x)
        print(f"\n5. LAST CONV LAYER:")
        print(f"   conv5_4: {last_features.shape}")

def demonstrate_no_skipping():
    """
    Demonstrate that no layers are actually skipped in computation
    """
    print("\n" + "=" * 55)
    print("DEMONSTRATING THAT NO LAYERS ARE SKIPPED")
    print("=" * 55)
    
    vgg = models.vgg19(pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    
    print("Forward pass through ALL layers in sequence:")
    current = x
    layer_count = 0
    conv_count = 0
    
    with torch.no_grad():
        for idx, layer in enumerate(vgg.features):
            current = layer(current)
            layer_count += 1
            
            if isinstance(layer, nn.Conv2d):
                conv_count += 1
                print(f"   Index {idx:2d}: Conv{conv_count:2d} -> Output shape: {current.shape}")
            else:
                layer_type = type(layer).__name__
                print(f"   Index {idx:2d}: {layer_type:10s} -> Output shape: {current.shape}")
    
    print(f"\nTotal layers processed: {layer_count}")
    print(f"Total conv layers: {conv_count}")
    print("Every single layer was computed - none were skipped!")

if __name__ == "__main__":
    print("COMPLETE VGG19 FEATURE EXTRACTOR")
    print("No layers skipped - all 16 conv layers available")
    print("=" * 55)
    
    compare_layer_selections()
    demonstrate_no_skipping()
    
    print("\n" + "=" * 55)
    print("USAGE EXAMPLES:")
    print("=" * 55)
    print("""
# Extract from ALL 16 convolutional layers:
extractor = CompleteVGG19FeatureExtractor()
all_features = extractor.get_all_conv_features(image)

# Extract from specific layers:
specific_features = extractor(image, ['conv1_1', 'conv3_2', 'conv5_4'])

# Extract by block:
block3_features = extractor.get_block_features(image, block_num=3)

# Traditional style transfer:
style_features = extractor.get_traditional_style_features(image)
content_features = extractor.get_content_features(image)
""")

