import torch
from torchvision import models

def analyze_vgg19_architecture():
    """
    Analyze the complete VGG19 architecture to understand layer indexing
    """
    vgg19 = models.vgg19(pretrained=True)
    
    print("COMPLETE VGG19 ARCHITECTURE ANALYSIS")
    print("=" * 60)
    
    print("\n1. FULL MODEL STRUCTURE:")
    print("-" * 30)
    print(vgg19)
    
    print("\n2. FEATURES MODULE DETAILED BREAKDOWN:")
    print("-" * 40)
    
    conv_count = 0
    relu_count = 0
    pool_count = 0
    
    for idx, layer in enumerate(vgg19.features):
        layer_type = type(layer).__name__
        
        if isinstance(layer, torch.nn.Conv2d):
            conv_count += 1
            conv_info = f"Conv{conv_count}: {layer.in_channels}->{layer.out_channels}, kernel={layer.kernel_size[0]}x{layer.kernel_size[1]}"
            print(f"Index {idx:2d}: {layer_type:15s} | {conv_info}")
        elif isinstance(layer, torch.nn.ReLU):
            relu_count += 1
            print(f"Index {idx:2d}: {layer_type:15s} | ReLU{relu_count}")
        elif isinstance(layer, torch.nn.MaxPool2d):
            pool_count += 1
            pool_info = f"MaxPool{pool_count}: kernel={layer.kernel_size}x{layer.kernel_size}, stride={layer.stride}"
            print(f"Index {idx:2d}: {layer_type:15s} | {pool_info}")
        else:
            print(f"Index {idx:2d}: {layer_type:15s}")
    
    print(f"\nTOTAL LAYERS: {len(vgg19.features)} layers")
    print(f"- Convolution layers: {conv_count}")
    print(f"- ReLU layers: {relu_count}")
    print(f"- MaxPool layers: {pool_count}")
    
    return vgg19

def show_commonly_used_layers():
    """
    Show which layers are commonly used in neural style transfer and why
    """
    print("\n3. NEURAL STYLE TRANSFER LAYER SELECTION:")
    print("-" * 45)
    
    # Standard layer mapping used in neural style transfer papers
    layer_mapping = {
        '0': ('conv1_1', 'Conv2d', 'First conv layer - captures low-level features'),
        '2': ('relu1_1', 'ReLU', 'After first conv (sometimes used)'),
        '5': ('conv2_1', 'Conv2d', 'After first pooling - medium-level features'),
        '7': ('relu2_1', 'ReLU', 'After conv2_1'),
        '10': ('conv3_1', 'Conv2d', 'After second pooling - higher-level features'),
        '12': ('relu3_1', 'ReLU', 'After conv3_1'),
        '14': ('conv3_2', 'Conv2d', 'Second conv in block 3'),
        '16': ('relu3_2', 'ReLU', 'After conv3_2'),
        '18': ('conv3_3', 'Conv2d', 'Third conv in block 3'),
        '19': ('conv4_1', 'Conv2d', 'After third pooling - very high-level features'),
        '21': ('conv4_2', 'Conv2d', 'CONTENT LAYER - most commonly used for content'),
        '23': ('relu4_2', 'ReLU', 'After conv4_2'),
        '25': ('conv4_3', 'Conv2d', 'Third conv in block 4'),
        '27': ('relu4_3', 'ReLU', 'After conv4_3'),
        '28': ('conv5_1', 'Conv2d', 'After fourth pooling - abstract features'),
        '30': ('conv5_2', 'Conv2d', 'LAST CONV LAYER'),
    }
    
    print("Index | Layer Name | Type    | Description")
    print("-" * 70)
    for idx, (name, layer_type, description) in layer_mapping.items():
        marker = " ★" if 'CONTENT' in description or 'commonly used' in description else ""
        marker += " ◆" if 'LAST' in description else ""
        print(f"{idx:5s} | {name:10s} | {layer_type:7s} | {description}{marker}")

def explain_layer_selection():
    """
    Explain why certain layers are selected and others are skipped
    """
    print("\n4. WHY ARE SOME LAYERS 'SKIPPED'?")
    print("-" * 35)
    
    print("""
REASON 1: ReLU layers are often skipped because:
- They don't add new information (just apply max(0, x))
- The conv layer output before ReLU often contains more rich information
- Some papers use pre-ReLU features, others use post-ReLU

REASON 2: Not all conv layers are equally useful:
- conv1_1 (idx 0): Good for fine textures and edges
- conv2_1 (idx 5): Good for medium-scale patterns  
- conv3_1 (idx 10): Good for larger patterns
- conv4_1 (idx 19): Good for high-level structures
- conv4_2 (idx 21): BEST for content representation
- conv5_1 (idx 28): Good for very abstract features

REASON 3: Multiple conv layers in same block:
- Block 3 has conv3_1, conv3_2, conv3_3, conv3_4
- Block 4 has conv4_1, conv4_2, conv4_3, conv4_4  
- Block 5 has conv5_1, conv5_2, conv5_3, conv5_4
- Usually only first 1-2 from each block are used

REASON 4: Computational efficiency:
- Using fewer layers = faster computation
- Selected layers provide good coverage of feature hierarchy
- Diminishing returns from using ALL layers
""")

def show_all_conv_layers():
    """
    Show ALL convolutional layers in VGG19 without skipping any
    """
    vgg19 = models.vgg19(pretrained=True)
    
    print("\n5. ALL CONVOLUTIONAL LAYERS (NO SKIPPING):")
    print("-" * 45)
    
    conv_layers = {}
    conv_count = 0
    
    for idx, layer in enumerate(vgg19.features):
        if isinstance(layer, torch.nn.Conv2d):
            conv_count += 1
            conv_layers[str(idx)] = f"conv{conv_count}"
            print(f"Index {idx:2d}: conv{conv_count} | {layer.in_channels:3d} -> {layer.out_channels:3d} channels")
    
    return conv_layers

def create_complete_feature_extractor():
    """
    Create a feature extractor that doesn't skip ANY convolutional layers
    """
    print("\n6. FEATURE EXTRACTOR WITH ALL CONV LAYERS:")
    print("-" * 45)
    
    code = '''
class CompleteVGG19FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = models.vgg19(pretrained=pretrained)
        self.features = vgg.features
        
        # Map ALL convolutional layers
        self.all_conv_layers = {
            '0': 'conv1_1',   '2': 'conv1_2',
            '5': 'conv2_1',   '7': 'conv2_2', 
            '10': 'conv3_1',  '12': 'conv3_2',  '14': 'conv3_3',  '16': 'conv3_4',
            '19': 'conv4_1',  '21': 'conv4_2',  '23': 'conv4_3',  '25': 'conv4_4',
            '28': 'conv5_1',  '30': 'conv5_2',  '32': 'conv5_3',  '34': 'conv5_4'
        }
        
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x, extract_all_conv=False):
        features = {}
        current_features = x
        
        for idx, layer in enumerate(self.features):
            current_features = layer(current_features)
            
            if extract_all_conv and str(idx) in self.all_conv_layers:
                layer_name = self.all_conv_layers[str(idx)]
                features[layer_name] = current_features
        
        return features
'''
    print(code)

if __name__ == "__main__":
    # Run complete analysis
    vgg19 = analyze_vgg19_architecture()
    show_commonly_used_layers()
    explain_layer_selection()
    all_conv_layers = show_all_conv_layers()
    create_complete_feature_extractor()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print("• VGG19 has 16 convolutional layers total")
    print("• Neural style transfer typically uses 5-6 strategically selected layers")
    print("• Layers are chosen to capture different levels of abstraction")
    print("• ReLU and pooling layers are usually skipped in feature extraction")
    print("• You can extract from ALL conv layers if needed (see code above)")

