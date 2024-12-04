import cv2
import numpy as np
from numpy.linalg import inv
from scipy.spatial.distance import cdist


def erase_small(image, THLD, invert=False):
    if invert:
        image = cv2.bitwise_not(image)
    image = np.uint8(image)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)

    # Step 3: Remove small components
    for i in range(1, num_labels):  # Skip the background label 0
        component_size = stats[i, cv2.CC_STAT_AREA]
        if component_size < THLD:
            # Paint small components black by setting pixels to 0
            image[labels == i] = 0
    return image

def distance2D(A, B):
    return ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2) ** 0.5

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


def calculate_hitmap_points(image_hitmap, final_centroids, dist_transform):
    """
    Calculate hitmap points for each centroid based on the intensity of pixels within a circle.

    Parameters:
    - image_hitmap: np.ndarray, the hitmap image.
    - final_centroids: list of tuples, each tuple is a (x, y) centroid.
    - dist_transform: np.ndarray, distance transform for determining the radius.

    Returns:
    - hitmap_points: list, intensity sums for each centroid.
    """
    # Initialize hitmap points
    hitmap_points = [0 for _ in range(len(final_centroids))]

    # Iterate through each centroid
    for k, centroid in enumerate(final_centroids):
        x, y = int(centroid[0]), int(centroid[1])
        radius = int(dist_transform[y, x])  # Radius determined by distance transform
        
        # Iterate over the pixels in the bounding box around the circle
        for i in range(max(0, x - radius), min(image_hitmap.shape[1], x + radius + 1)):
            for j in range(max(0, y - radius), min(image_hitmap.shape[0], y + radius + 1)):
                # Check if the pixel (i, j) is within the circle
                if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                    # Add intensity of the pixel to hitmap points for this centroid
                    hitmap_points[k] += image_hitmap[j, i]

    return hitmap_points

def gen_circles(image, hitmap):
    '''
    Recieves image and hitmap image, returns the output image with XAI results included

    Parameters:
    - image, numpy.ndarray
    - hitmap, numpy.ndarray

    Returns:
    - original image with XAI results added, numpy.ndarray
    '''
    h, w, _ = image.shape

    x1, y1 = 200, 10  # Top-left corner
    x2, y2 = 1500, 920  # Bottom-right corner

    # before cropping the hitmap, enlarge hitmap to match the size of image
    hitmap = cv2.resize(hitmap, (w, h), interpolation=cv2.INTER_AREA)
    hitmap = hitmap[y1:y2, x1:x2]
    hitmap = cv2.cvtColor(hitmap, cv2.COLOR_BGR2GRAY)

    # Step 3: Crop the image using slicing
    image = image[y1:y2, x1:x2]
    image_original = image

    # Convert the image to float32 for precision in calculations (optional but recommended)
    image = np.float32(image)

    # Step 1: Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    cv2.imwrite('hsv1.jpg', image)

    # Step 3: Create a mask using the HSV range
    h, s, v = cv2.split(hsv_image)
    mask1 = 10 < h
    mask2 = h < 35

    # Step 4: Apply the mask to retain only the pixels within the specified hue range
    result = np.logical_and(mask1, mask2)
    image[result == 0] = [0,0,0]

    # Apply thresh to erase small white regions
    THRSH = 50000
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.uint8(gray)

    # erase small components
    image = erase_small(image, THRSH, False)
    image = erase_small(image, THRSH, True)

    # apply erosion and dilation
    k = np.ones((20,20),np.uint8)
    image = cv2.erode(image, k, 2)
    k = np.ones((25,25),np.uint8)
    image = cv2.dilate(image, k, 16)
    result_image = erase_small(image, THRSH, True)

    # Pad the image to limit distance calculations at boundaries
    padding = 10  # Adjust as necessary
    padded_image = cv2.copyMakeBorder(result_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

    # Apply distance transform on the padded image
    dist_transform_padded = cv2.distanceTransform(padded_image, cv2.DIST_L2, 5)
    dist_transform = dist_transform_padded[padding:-padding, padding:-padding]

    # threshold를 올려가면서 thresholding을 하면서, 없어질 때마다 기록
    sure_fg_old = None
    numLabels_old = None
    markers_old = None 
    stats_old = None
    centroids_old = None

    output_centroids = []
    final_centroids = []

    for t in range(100):
        thld = 0.1 + 0.01 * t
        if t != 0:
            sure_fg_old = sure_fg
            numLabels_old, markers_old, stats_old, centroids_old = numLabels, markers, stats, centroids

        _, sure_fg = cv2.threshold(dist_transform, thld * dist_transform.max(),255,0)
        sure_fg = np.uint8(sure_fg)
        numLabels, markers, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)

        # Initialize a list to mark components for deletion
        components_to_delete = []

        # Loop over all components except the background (0th label)
        for i in range(1, numLabels):
            size = stats[i, cv2.CC_STAT_AREA]
            
            # Check if component is small
            if size < 10:
                # Get coordinates of all pixels in the small component
                small_component_pixels = np.column_stack(np.where(markers == i))
                
                # Loop through other components to find the nearest large one
                for j in range(1, numLabels):
                    if i != j and stats[j, cv2.CC_STAT_AREA] > 100:
                        # Get coordinates of all pixels in the larger component
                        large_component_pixels = np.column_stack(np.where(markers == j))
                        
                        # Compute pairwise distances between pixels in the two components
                        distances = cdist(small_component_pixels, large_component_pixels)
                        min_distance = np.min(distances)
                        
                        # Check if the nearest large component is within the distance threshold
                        if min_distance < 3:
                            components_to_delete.append(i)
                            break  # Stop checking further components once criteria are met

        # Set markers for components to delete to zero (background)
        for i in components_to_delete:
            markers[markers == i] = 0

        # Re-label the markers to update the component indices after deletion
        numLabels, markers, stats, centroids = cv2.connectedComponentsWithStats((markers > 0).astype(np.uint8))

        if t!= 0:
            old_components = [1 for _ in range(numLabels_old)]
            for i in range(1, numLabels):
                y, x = int(centroids[i][1]), int(centroids[i][0])
                if sure_fg_old[y, x] > 0:
                    component_label = markers_old[y, x]
                    if component_label > 0:
                        old_components[component_label] = 0
            for j in range(1, numLabels_old):
                if old_components[j] == 1:
                    output_centroids.append(centroids_old[j])
                    # print("centroid added : ", centroids_old[j], "t = ", t, "j = ", j)

    # implement non maximum suppression
    # rule: for all centroids in output_centroids, repeat:
    #   take the centroid with maximum radius, A, and put it in final_centroids
    #   for all other centroid B, if 0.3 * min(radius(A), radius(B)) > L2 distance(A, B):
    #       erase B from output_centroids
    # note that radius(centroid) = dist_transform[y, x], with x, y = int(centroid[0]), int(centroid[1])

    # Repeat until output_centroids is empty
    while output_centroids:
        # Find the centroid with the maximum radius
        max_index = max(range(len(output_centroids)), key=lambda i: dist_transform[int(output_centroids[i][1]), int(output_centroids[i][0])])
        A = output_centroids[max_index]

        # Add the centroid with the maximum radius to final_centroids
        final_centroids.append(A)

        # Get radius of A
        radius_A = dist_transform[int(A[1]), int(A[0])]

        # Remove the selected centroid from output_centroids
        output_centroids = [output_centroids[i] for i in range(len(output_centroids)) if i != max_index]

        # Remove centroids that overlap with A based on the given rule
        output_centroids = [
            B for B in output_centroids
            if 1.1 * min(radius_A, dist_transform[int(B[1]), int(B[0])]) <= distance2D(A, B)
            and radius_A - distance2D(A, B) + dist_transform[int(B[1]), int(B[0])] 
            <= 0.8 * dist_transform[int(B[1]), int(B[0])]
        ]
    
    erase_indexed = []
    for i in range(len(final_centroids)):
        centroid = final_centroids[i]
        x, y = int(centroid[0]), int(centroid[1])
        radius = int(dist_transform[y, x])
        if radius < 10:
            #erase centroid from centroids
            erase_indexed.append(i)
    
    # Remove centroids whose indices are in erase_indexed
    final_centroids = [final_centroids[j] for j in range(len(final_centroids)) if j not in erase_indexed]
    radii = []
    filtered_points = []

    for ind in range(len(final_centroids)):
        centroid = final_centroids[ind]
        x, y = int(centroid[0]), int(centroid[1])
        radius = int(dist_transform[y, x])
        radii.append(radius * 1.4)

    # 후보 circle들을 사용해 크기가 큰 connected component, 나머지 버리기
    height, width, _ = image_original.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for h in range(len(final_centroids)):
        x, y = int(final_centroids[h][0]), int(final_centroids[h][1])
        radius = int(radii[h])
        print(x, y, radius)
        cv2.circle(mask, (x, y), radius, 1, -1)  # Draw filled circles with value 1

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    sizes = stats[:, cv2.CC_STAT_AREA]
    largest_size = max(sizes[1:])
    # valid components = 가장 큰 component + 그 크기의 50% 이상의 크기인 component들
    valid_labels = [i for i in range(1, num_labels) if sizes[i] >= 0.5 * largest_size]
    filtered_mask = np.isin(labels, valid_labels).astype(np.uint8)

    for x, y in final_centroids:
        # Ensure the point is within the image bounds
        if 0 <= int(round(y)) < filtered_mask.shape[0] and 0 <= int(round(x)) < filtered_mask.shape[1]:
            if filtered_mask[int(round(y)), int(round(x))] == 1:
                filtered_points.append((x, y))
    # final_centroids update
    final_centroids = filtered_points

    # hitmap 추출
    hitmap_points = calculate_hitmap_points(hitmap, final_centroids, dist_transform)
    ind = np.argmax(np.array(hitmap_points))

    # 치아 후보인 circle들을 전부 확인하고 싶으면 아래 주석 지우면 됨
    '''
    # add all circles to the image
    for centroid in final_centroids:
        x, y = int(centroid[0]), int(centroid[1])
        radius = int(dist_transform[y, x])
        if radius > 9:
            cv2.circle(image_original, (x, y), radius, (0, 255, 0), 2)  # Green circle with 2-pixel thickness
    '''

    # XAI 상에서 가장 점수가 높은 치아 표시
    centroid = final_centroids[ind]
    x, y = int(centroid[0]), int(centroid[1])
    radius = int(dist_transform[y, x])
    if radius > 9:
        cv2.circle(image_original, (x, y), radius, (0, 255, 0), 2)  # Green circle with 2-pixel thickness
    
    return image_original




# example usage
image = cv2.imread('path/to/photo.jpg')
hitmap = cv2.imread('path/to/corresponding/hitmap.jpg')

image_with_circles = gen_circles(image, hitmap)

cv2.imshow("Image with XAI", image_with_circles.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()