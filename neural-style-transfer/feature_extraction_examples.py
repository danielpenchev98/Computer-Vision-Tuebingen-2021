import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# Load VGG19 model
cnn = models.vgg19(pretrained=True)
cnn.eval()  # Set to evaluation mode

print("VGG19 Architecture:")
print(cnn)
print("\n" + "="*50)

# Method 1: Get output of the last convolutional layer (before classifier)
def get_last_conv_features_method1(model, x):
    """
    Method 1: Use the features module which contains all convolutional layers
    The last feature layer is the last layer in model.features
    """
    # VGG19 features module contains all conv layers + pooling + ReLU
    features = model.features(x)  # Shape: [batch, 512, H, W]
    return features

# Method 2: Extract features layer by layer and get the last one
def get_last_conv_features_method2(model, x):
    """
    Method 2: Iterate through features and get output of last layer
    """
    features = x
    for layer in model.features:
        features = layer(features)
    return features

# Method 3: Use forward hooks to capture intermediate outputs
class FeatureExtractor:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.features = None
        self.hook = target_layer.register_forward_hook(self.save_features)
    
    def save_features(self, module, input, output):
        self.features = output
    
    def extract(self, x):
        self.model(x)
        return self.features
    
    def remove_hook(self):
        self.hook.remove()

def get_last_conv_features_method3(model, x):
    """
    Method 3: Use hooks to extract features from specific layers
    """
    # Get the last convolutional layer
    last_conv_layer = None
    for module in model.features.modules():
        if isinstance(module, nn.Conv2d):
            last_conv_layer = module
    
    if last_conv_layer is None:
        raise ValueError("No convolutional layer found")
    
    extractor = FeatureExtractor(model, last_conv_layer)
    features = extractor.extract(x)
    extractor.remove_hook()
    return features

# Method 4: Create a custom model that outputs features
class VGG19Features(nn.Module):
    def __init__(self, pretrained_vgg):
        super(VGG19Features, self).__init__()
        self.features = pretrained_vgg.features
        
    def forward(self, x):
        return self.features(x)

def get_last_conv_features_method4(model, x):
    """
    Method 4: Create a wrapper model that only has the features part
    """
    feature_model = VGG19Features(model)
    return feature_model(x)

# Method 5: Get features from specific named layers (for neural style transfer)
def get_features_from_layers(model, x, layers):
    """
    Method 5: Extract features from multiple specific layers
    Useful for neural style transfer where you need features from different depths
    """
    features = {}
    current_features = x
    
    layer_names = {
        '0': 'conv1_1',   # First conv layer
        '5': 'conv2_1',   # After first pooling
        '10': 'conv3_1',  # After second pooling
        '19': 'conv4_1',  # After third pooling
        '21': 'conv4_2',  # Common for content loss
        '28': 'conv5_1',  # After fourth pooling
        '30': 'conv5_2',  # Last conv layer before final pooling
    }
    
    for idx, layer in enumerate(model.features):
        current_features = layer(current_features)
        if str(idx) in layer_names and str(idx) in layers:
            features[layer_names[str(idx)]] = current_features
    
    return features

# Method 6: Get the absolute last feature before classification
def get_final_features_before_classifier(model, x):
    """
    Method 6: Get features right before the classifier (after adaptive pooling)
    This gives you the 1D feature vector that goes into the classifier
    """
    # Get conv features
    conv_features = model.features(x)  # [batch, 512, H, W]
    
    # Apply adaptive pooling (like VGG does before classifier)
    pooled_features = model.avgpool(conv_features)  # [batch, 512, 7, 7]
    
    # Flatten for classifier input
    flattened = torch.flatten(pooled_features, 1)  # [batch, 512*7*7]
    
    return {
        'conv_features': conv_features,      # Last conv output
        'pooled_features': pooled_features,  # After adaptive pooling
        'flattened': flattened              # Flattened for classifier
    }

# Example usage and testing
if __name__ == "__main__":
    # Create a sample input
    batch_size = 1
    channels = 3
    height = width = 224  # VGG expects 224x224 input
    
    sample_input = torch.randn(batch_size, channels, height, width)
    
    print("Testing different methods to extract features:\n")
    
    # Test Method 1
    features1 = get_last_conv_features_method1(cnn, sample_input)
    print(f"Method 1 - Last conv features shape: {features1.shape}")
    
    # Test Method 2  
    features2 = get_last_conv_features_method2(cnn, sample_input)
    print(f"Method 2 - Last conv features shape: {features2.shape}")
    
    # Test Method 3
    features3 = get_last_conv_features_method3(cnn, sample_input)
    print(f"Method 3 - Last conv features shape: {features3.shape}")
    
    # Test Method 4
    features4 = get_last_conv_features_method4(cnn, sample_input)
    print(f"Method 4 - Last conv features shape: {features4.shape}")
    
    # Test Method 5 - Get multiple layers for style transfer
    style_layers = ['0', '5', '10', '19', '28']
    content_layers = ['21']
    all_layers = style_layers + content_layers
    
    multi_features = get_features_from_layers(cnn, sample_input, all_layers)
    print(f"\nMethod 5 - Multiple layer features:")
    for layer_name, features in multi_features.items():
        print(f"  {layer_name}: {features.shape}")
    
    # Test Method 6
    final_features = get_final_features_before_classifier(cnn, sample_input)
    print(f"\nMethod 6 - Final features before classifier:")
    for key, features in final_features.items():
        print(f"  {key}: {features.shape}")
    
    # Verify all methods give the same result for conv features
    print(f"\nVerification - All methods return same conv features: {torch.allclose(features1, features2, features3, features4)}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print("1. For neural style transfer: Use Method 5 to get features from multiple layers")
    print("2. For feature extraction: Use Method 1 (simplest)")
    print("3. For custom architectures: Use Method 3 (hooks)")
    print("4. For classification features: Use Method 6")

