import cv2
import numpy as np
import matplotlib.pyplot as plt
from blended_stitching import (
    stich_images_blended, 
    stich_images_simple_blended, 
    stich_images_vectorized_blended,
    stich_images_gradient_blended
)

def load_sample_images():
    """
    Load sample images for testing the blending functions.
    You can replace these with your own images.
    """
    # Try to load images from the current directory
    try:
        img1 = cv2.imread('image-1.jpg')
        img2 = cv2.imread('image-2.jpg')
        if img1 is None or img2 is None:
            raise FileNotFoundError("Sample images not found")
    except:
        # Create synthetic images for demonstration
        print("Creating synthetic images for demonstration...")
        img1 = np.zeros((300, 400, 3), dtype=np.uint8)
        img2 = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Create a gradient pattern for img1
        for i in range(300):
            for j in range(400):
                img1[i, j] = [i//2, j//2, 128]
        
        # Create a different pattern for img2
        for i in range(300):
            for j in range(400):
                img2[i, j] = [128, i//2, j//2]
    
    return img1, img2

def create_sample_homography():
    """
    Create a sample homography matrix for demonstration.
    In practice, you would compute this from feature matching.
    """
    # Simple translation homography
    H = np.array([
        [1.0, 0.0, 100],  # Translate 100 pixels to the right
        [0.0, 1.0, 50],   # Translate 50 pixels down
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    return H

def compare_blending_methods(img1, img2, H):
    """
    Compare different blending methods and visualize results.
    """
    print("Testing different blending methods...")
    
    # Test all blending methods
    methods = {
        'Original (no blending)': lambda: stich_images_blended(img1, img2, H),
        'Simple Blended': lambda: stich_images_simple_blended(img1, img2, H),
        'Vectorized Blended': lambda: stich_images_vectorized_blended(img1, img2, H),
        'Gradient Blended': lambda: stich_images_gradient_blended(img1, img2, H)
    }
    
    results = {}
    for method_name, method_func in methods.items():
        print(f"Processing {method_name}...")
        try:
            results[method_name] = method_func()
        except Exception as e:
            print(f"Error in {method_name}: {e}")
            results[method_name] = None
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Show original images
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Image 1 (Reference)')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Image 2 (To be warped)')
    axes[1].axis('off')
    
    # Show results
    for i, (method_name, result) in enumerate(results.items()):
        if result is not None:
            axes[i+2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            axes[i+2].set_title(method_name)
        else:
            axes[i+2].text(0.5, 0.5, f'Error in {method_name}', 
                          ha='center', va='center', transform=axes[i+2].transAxes)
        axes[i+2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results

def test_with_real_images():
    """
    Test the blending functions with real images if available.
    """
    print("Testing with real images...")
    
    # Load images
    img1, img2 = load_sample_images()
    
    # Create homography (in practice, you'd compute this from feature matching)
    H = create_sample_homography()
    
    # Compare methods
    results = compare_blending_methods(img1, img2, H)
    
    # Save results
    for method_name, result in results.items():
        if result is not None:
            filename = f"stitched_{method_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.jpg"
            cv2.imwrite(filename, result)
            print(f"Saved {filename}")

def demonstrate_weight_calculation():
    """
    Demonstrate how the weight calculation works.
    """
    print("Demonstrating weight calculation...")
    
    # Create a simple test image
    test_img = np.zeros((100, 100), dtype=np.uint8)
    test_img[25:75, 25:75] = 255
    
    # Calculate weights
    from blended_stitching import get_pixel_weights
    weights = get_pixel_weights(test_img)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(test_img, cmap='gray')
    axes[0].set_title('Test Image')
    axes[0].axis('off')
    
    im = axes[1].imshow(weights, cmap='viridis')
    axes[1].set_title('Pixel Weights (higher = closer to center)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Image Stitching with Blending Demo")
    print("=" * 40)
    
    # Demonstrate weight calculation
    demonstrate_weight_calculation()
    
    # Test with images
    test_with_real_images()
    
    print("\nDemo completed! Check the generated images to see the blending results.") 