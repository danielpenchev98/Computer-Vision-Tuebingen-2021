import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class VGG19FeatureExtractor(nn.Module):
    """
    VGG19 Feature Extractor for Neural Style Transfer
    Extracts features from multiple layers commonly used in style transfer
    """
    
    def __init__(self, pretrained=True):
        super(VGG19FeatureExtractor, self).__init__()
        
        # Load pretrained VGG19
        vgg = models.vgg19(pretrained=pretrained)
        self.features = vgg.features
        
        # Define layer mappings for style transfer
        # These are the commonly used layers in neural style transfer papers
        self.layer_names = {
            '0': 'conv1_1',    # Style layer 1
            '5': 'conv2_1',    # Style layer 2  
            '10': 'conv3_1',   # Style layer 3
            '19': 'conv4_1',   # Style layer 4
            '21': 'conv4_2',   # Content layer (commonly used)
            '28': 'conv5_1',   # Style layer 5
            '30': 'conv5_2',   # Last conv layer
        }
        
        # Default layers for content and style
        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        
        # Freeze parameters (don't train the VGG network)
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x, layers=None):
        """
        Forward pass to extract features
        
        Args:
            x: Input tensor [batch_size, 3, H, W]
            layers: List of layer names to extract. If None, extracts all default layers
            
        Returns:
            Dictionary with layer names as keys and feature tensors as values
        """
        if layers is None:
            layers = list(self.layer_names.values())
        
        features = {}
        current_features = x
        
        for idx, layer in enumerate(self.features):
            current_features = layer(current_features)
            
            # Check if this layer should be saved
            layer_name = self.layer_names.get(str(idx))
            if layer_name and layer_name in layers:
                features[layer_name] = current_features
        
        return features
    
    def get_content_features(self, x):
        """Get features from content layers only"""
        return self.forward(x, self.content_layers)
    
    def get_style_features(self, x):
        """Get features from style layers only"""
        return self.forward(x, self.style_layers)
    
    def get_last_conv_features(self, x):
        """Get output from the last convolutional layer"""
        return self.forward(x, ['conv5_2'])['conv5_2']

def get_gram_matrix(features):
    """
    Calculate Gram matrix for style loss
    
    Args:
        features: Feature tensor [batch, channels, height, width]
        
    Returns:
        Gram matrix [batch, channels, channels]
    """
    batch_size, channels, height, width = features.size()
    
    # Reshape features to [batch, channels, height*width]
    features = features.view(batch_size, channels, height * width)
    
    # Calculate Gram matrix: G = F * F^T
    gram_matrix = torch.bmm(features, features.transpose(1, 2))
    
    # Normalize by number of elements
    gram_matrix = gram_matrix / (channels * height * width)
    
    return gram_matrix

# Example usage for neural style transfer
def neural_style_example():
    """
    Example of how to use the feature extractor for neural style transfer
    """
    
    # Initialize the feature extractor
    feature_extractor = VGG19FeatureExtractor(pretrained=True)
    feature_extractor.eval()
    
    # Create sample images (content, style, generated)
    batch_size = 1
    channels = 3
    height = width = 256
    
    content_img = torch.randn(batch_size, channels, height, width)
    style_img = torch.randn(batch_size, channels, height, width)
    generated_img = torch.randn(batch_size, channels, height, width, requires_grad=True)
    
    print("Extracting features for neural style transfer...")
    
    # Extract content features
    content_features = feature_extractor.get_content_features(content_img)
    print(f"Content features extracted from layers: {list(content_features.keys())}")
    for layer, features in content_features.items():
        print(f"  {layer}: {features.shape}")
    
    # Extract style features and compute Gram matrices
    style_features = feature_extractor.get_style_features(style_img)
    style_grams = {}
    print(f"\nStyle features extracted from layers: {list(style_features.keys())}")
    for layer, features in style_features.items():
        style_grams[layer] = get_gram_matrix(features)
        print(f"  {layer}: {features.shape} -> Gram: {style_grams[layer].shape}")
    
    # Extract features from generated image
    generated_content_features = feature_extractor.get_content_features(generated_img)
    generated_style_features = feature_extractor.get_style_features(generated_img)
    
    # Calculate content loss
    content_loss = 0
    for layer in content_features:
        content_loss += F.mse_loss(generated_content_features[layer], content_features[layer])
    
    # Calculate style loss
    style_loss = 0
    for layer in style_features:
        generated_gram = get_gram_matrix(generated_style_features[layer])
        style_loss += F.mse_loss(generated_gram, style_grams[layer])
    
    print(f"\nLoss calculation:")
    print(f"Content Loss: {content_loss.item():.6f}")
    print(f"Style Loss: {style_loss.item():.6f}")
    
    # Get last convolutional layer features
    last_conv_features = feature_extractor.get_last_conv_features(generated_img)
    print(f"\nLast conv layer features shape: {last_conv_features.shape}")
    
    return feature_extractor, content_loss, style_loss

# Simple function to get just the last feature layer output
def get_last_feature_layer_output(image):
    """
    Simple function to get the output of the last feature layer from VGG19
    
    Args:
        image: Input tensor [batch_size, 3, H, W]
        
    Returns:
        Features from the last convolutional layer
    """
    # Load VGG19
    vgg19 = models.vgg19(pretrained=True)
    vgg19.eval()
    
    # Extract features (this includes all conv layers, pooling, and ReLU)
    with torch.no_grad():
        features = vgg19.features(image)
    
    return features

if __name__ == "__main__":
    print("VGG19 Feature Extraction for Neural Style Transfer")
    print("=" * 55)
    
    # Run the neural style example
    feature_extractor, content_loss, style_loss = neural_style_example()
    
    print("\n" + "=" * 55)
    print("QUICK USAGE:")
    print("=" * 55)
    print("# For simple last layer extraction:")
    print("last_features = get_last_feature_layer_output(your_image)")
    print()
    print("# For neural style transfer:")
    print("extractor = VGG19FeatureExtractor()")
    print("content_features = extractor.get_content_features(content_img)")
    print("style_features = extractor.get_style_features(style_img)")
    print("last_conv = extractor.get_last_conv_features(img)")

