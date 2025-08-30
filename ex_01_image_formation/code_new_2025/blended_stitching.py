import numpy as np
import cv2

def get_pixel_weights(img):
    """
    Calculate pixel weights based on distance from image boundaries.
    Pixels closer to the center get higher weights.
    
    Args:
        img (array): Input image
        
    Returns:
        array: Weight matrix with same shape as input image
    """
    rows, cols = img.shape[:2]
    img_pixel_weights = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            img_pixel_weights[i, j] = min(i, rows-i, j, cols-j)
    return img_pixel_weights

def stich_images_blended(img1, img2, H):
    ''' 
    Stitches together the images via given homography H with blending.
    Uses the formula: (w1*I1 + w2*I2) / (w1 + w2) to blend overlapping regions.

    Args:
        img1 (array): First image (reference image)
        img2 (array): Second image (to be warped)
        H (array): Homography matrix
        
    Returns:
        array: Blended stitched image
    '''
    
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Define corners of both images
    list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)

    # Transform corners of img2 using homography
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

    # Calculate bounding box for the output image
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min,-y_min]

    # Create translation matrix
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    
    # Warp img2 to the output coordinate system
    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    
    # Create weight matrices for both images
    # For img1, we need to create weights in the output coordinate system
    img1_weights = get_pixel_weights(img1)
    img1_weights_padded = np.zeros((y_max-y_min, x_max-x_min))
    img1_weights_padded[translation_dist[1]:rows1+translation_dist[1], 
                       translation_dist[0]:cols1+translation_dist[0]] = img1_weights
    
    # For img2, we need to warp the weights using the same transformation
    img2_weights = get_pixel_weights(img2)
    img2_weights_warped = cv2.warpPerspective(img2_weights, H_translation.dot(H), 
                                             (x_max-x_min, y_max-y_min))
    
    # Handle the case where img2_weights_warped might be single channel
    if len(img2_weights_warped.shape) == 2:
        img2_weights_warped = img2_weights_warped[..., np.newaxis]
    if len(img1_weights_padded.shape) == 2:
        img1_weights_padded = img1_weights_padded[..., np.newaxis]
    
    # Create the blended output
    blended_output = np.zeros_like(output_img, dtype=np.float32)
    
    # Apply blending formula: (w1*I1 + w2*I2) / (w1 + w2)
    # First, create the img1 in output coordinate system
    img1_padded = np.zeros_like(output_img, dtype=np.float32)
    img1_padded[translation_dist[1]:rows1+translation_dist[1], 
                translation_dist[0]:cols1+translation_dist[0]] = img1.astype(np.float32)
    
    # Calculate weighted sum
    numerator = img1_weights_padded * img1_padded + img2_weights_warped * output_img.astype(np.float32)
    denominator = img1_weights_padded + img2_weights_warped
    
    # Avoid division by zero
    denominator[denominator == 0] = 1
    
    blended_output = numerator / denominator
    
    # Convert back to uint8
    blended_output = np.clip(blended_output, 0, 255).astype(np.uint8)
    
    return blended_output

def stich_images_simple_blended(img1, img2, H):
    '''
    Simplified version of blended stitching that uses a more straightforward approach.
    This version creates a mask for the overlapping region and blends only there.
    
    Args:
        img1 (array): First image (reference image)
        img2 (array): Second image (to be warped)
        H (array): Homography matrix
        
    Returns:
        array: Blended stitched image
    '''
    
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Define corners of both images
    list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)

    # Transform corners of img2 using homography
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

    # Calculate bounding box for the output image
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min,-y_min]

    # Create translation matrix
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    
    # Warp img2 to the output coordinate system
    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    
    # Create the output image starting with img1
    final_img = np.zeros((y_max-y_min, x_max-x_min, img1.shape[2]), dtype=np.uint8)
    final_img[translation_dist[1]:rows1+translation_dist[1], 
              translation_dist[0]:cols1+translation_dist[0]] = img1
    
    # Create a mask for the overlapping region
    overlap_mask = np.zeros((y_max-y_min, x_max-x_min), dtype=np.uint8)
    overlap_mask[translation_dist[1]:rows1+translation_dist[1], 
                translation_dist[0]:cols1+translation_dist[0]] = 255
    
    # Create a mask for where img2 has valid pixels (non-zero)
    img2_mask = np.zeros((y_max-y_min, x_max-x_min), dtype=np.uint8)
    if len(output_img.shape) == 3:
        img2_valid = np.any(output_img > 0, axis=2)
    else:
        img2_valid = output_img > 0
    img2_mask[img2_valid] = 255
    
    # Find overlapping pixels
    overlap_region = cv2.bitwise_and(overlap_mask, img2_mask)
    overlap_coords = np.where(overlap_region > 0)
    
    # Create weight matrices for blending
    img1_weights = get_pixel_weights(img1)
    img2_weights = get_pixel_weights(img2)
    img2_weights_warped = cv2.warpPerspective(img2_weights, H_translation.dot(H), 
                                             (x_max-x_min, y_max-y_min))
    
    # Blend only in the overlapping region
    for y, x in zip(overlap_coords[0], overlap_coords[1]):
        # Get corresponding coordinates in img1
        img1_y = y - translation_dist[1]
        img1_x = x - translation_dist[0]
        
        if 0 <= img1_y < rows1 and 0 <= img1_x < cols1:
            w1 = img1_weights[img1_y, img1_x]
            w2 = img2_weights_warped[y, x]
            
            # Apply blending formula
            if w1 + w2 > 0:
                blended_pixel = (w1 * final_img[y, x].astype(np.float32) + 
                               w2 * output_img[y, x].astype(np.float32)) / (w1 + w2)
                final_img[y, x] = np.clip(blended_pixel, 0, 255).astype(np.uint8)
    
    # Copy non-overlapping regions from img2
    non_overlap_mask = cv2.bitwise_and(img2_mask, cv2.bitwise_not(overlap_mask))
    final_img[non_overlap_mask > 0] = output_img[non_overlap_mask > 0]
    
    return final_img

def stich_images_vectorized_blended(img1, img2, H):
    '''
    Vectorized version of blended stitching for better performance.
    Uses the formula: (w1*I1 + w2*I2) / (w1 + w2) with vectorized operations.
    
    Args:
        img1 (array): First image (reference image)
        img2 (array): Second image (to be warped)
        H (array): Homography matrix
        
    Returns:
        array: Blended stitched image
    '''
    
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Define corners of both images
    list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)

    # Transform corners of img2 using homography
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

    # Calculate bounding box for the output image
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min,-y_min]

    # Create translation matrix
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    
    # Warp img2 to the output coordinate system
    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    
    # Create weight matrices for both images
    img1_weights = get_pixel_weights(img1)
    img2_weights = get_pixel_weights(img2)
    img2_weights_warped = cv2.warpPerspective(img2_weights, H_translation.dot(H), 
                                             (x_max-x_min, y_max-y_min))
    
    # Create img1 in output coordinate system
    img1_padded = np.zeros((y_max-y_min, x_max-x_min, img1.shape[2]), dtype=np.float32)
    img1_padded[translation_dist[1]:rows1+translation_dist[1], 
                translation_dist[0]:cols1+translation_dist[0]] = img1.astype(np.float32)
    
    # Create img1 weights in output coordinate system
    img1_weights_padded = np.zeros((y_max-y_min, x_max-x_min), dtype=np.float32)
    img1_weights_padded[translation_dist[1]:rows1+translation_dist[1], 
                       translation_dist[0]:cols1+translation_dist[0]] = img1_weights
    
    # Create masks for valid regions
    img1_valid = img1_weights_padded > 0
    img2_valid = img2_weights_warped > 0
    
    # Initialize output with img1
    final_img = img1_padded.copy()
    
    # Create blending mask (where both images have valid pixels)
    blend_mask = img1_valid & img2_valid
    
    # Apply vectorized blending where both images overlap
    if blend_mask.any():
        # Expand weights to match image dimensions for broadcasting
        w1_expanded = img1_weights_padded[..., np.newaxis]
        w2_expanded = img2_weights_warped[..., np.newaxis]
        
        # Apply blending formula only where both images are valid
        numerator = w1_expanded * img1_padded + w2_expanded * output_img.astype(np.float32)
        denominator = w1_expanded + w2_expanded
        
        # Avoid division by zero
        denominator[denominator == 0] = 1
        
        blended_region = numerator / denominator
        
        # Apply blending only in the overlap region
        final_img[blend_mask] = blended_region[blend_mask]
    
    # Copy img2 pixels where img1 is not valid but img2 is
    img2_only_mask = ~img1_valid & img2_valid
    final_img[img2_only_mask] = output_img[img2_only_mask]
    
    # Convert back to uint8
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    
    return final_img

def stich_images_gradient_blended(img1, img2, H, blend_width=50):
    '''
    Advanced blending using gradient-based weights for smoother transitions.
    Creates a gradual transition in the overlapping region.
    
    Args:
        img1 (array): First image (reference image)
        img2 (array): Second image (to be warped)
        H (array): Homography matrix
        blend_width (int): Width of the blending region in pixels
        
    Returns:
        array: Blended stitched image
    '''
    
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Define corners of both images
    list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)

    # Transform corners of img2 using homography
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

    # Calculate bounding box for the output image
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min,-y_min]

    # Create translation matrix
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    
    # Warp img2 to the output coordinate system
    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    
    # Create img1 in output coordinate system
    img1_padded = np.zeros((y_max-y_min, x_max-x_min, img1.shape[2]), dtype=np.float32)
    img1_padded[translation_dist[1]:rows1+translation_dist[1], 
                translation_dist[0]:cols1+translation_dist[0]] = img1.astype(np.float32)
    
    # Create gradient-based weights for img1
    img1_weights = np.zeros((y_max-y_min, x_max-x_min), dtype=np.float32)
    img1_region = img1_weights[translation_dist[1]:rows1+translation_dist[1], 
                              translation_dist[0]:cols1+translation_dist[0]]
    
    # Create gradient weights (stronger towards the center of img1)
    for i in range(img1_region.shape[0]):
        for j in range(img1_region.shape[1]):
            # Distance from center of img1
            center_i, center_j = rows1 // 2, cols1 // 2
            dist_from_center = np.sqrt((i - center_i)**2 + (j - center_j)**2)
            max_dist = np.sqrt(center_i**2 + center_j**2)
            img1_region[i, j] = 1.0 - (dist_from_center / max_dist)
    
    # Create gradient-based weights for img2 (warped)
    img2_weights = np.zeros((rows2, cols2), dtype=np.float32)
    for i in range(rows2):
        for j in range(cols2):
            # Distance from center of img2
            center_i, center_j = rows2 // 2, cols2 // 2
            dist_from_center = np.sqrt((i - center_i)**2 + (j - center_j)**2)
            max_dist = np.sqrt(center_i**2 + center_j**2)
            img2_weights[i, j] = 1.0 - (dist_from_center / max_dist)
    
    img2_weights_warped = cv2.warpPerspective(img2_weights, H_translation.dot(H), 
                                             (x_max-x_min, y_max-y_min))
    
    # Create masks for valid regions
    img1_valid = img1_weights > 0
    img2_valid = img2_weights_warped > 0
    
    # Initialize output with img1
    final_img = img1_padded.copy()
    
    # Create blending mask (where both images have valid pixels)
    blend_mask = img1_valid & img2_valid
    
    # Apply gradient-based blending where both images overlap
    if blend_mask.any():
        # Expand weights to match image dimensions for broadcasting
        w1_expanded = img1_weights[..., np.newaxis]
        w2_expanded = img2_weights_warped[..., np.newaxis]
        
        # Apply blending formula only where both images are valid
        numerator = w1_expanded * img1_padded + w2_expanded * output_img.astype(np.float32)
        denominator = w1_expanded + w2_expanded
        
        # Avoid division by zero
        denominator[denominator == 0] = 1
        
        blended_region = numerator / denominator
        
        # Apply blending only in the overlap region
        final_img[blend_mask] = blended_region[blend_mask]
    
    # Copy img2 pixels where img1 is not valid but img2 is
    img2_only_mask = ~img1_valid & img2_valid
    final_img[img2_only_mask] = output_img[img2_only_mask]
    
    # Convert back to uint8
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    
    return final_img 