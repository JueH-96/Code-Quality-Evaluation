def share_face(cuboid1, cuboid2):
    x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = cuboid1
    x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = cuboid2
    
    # Check if they share a face along the X-axis
    if x1_max == x2_min or x1_min == x2_max:
        # Check if the Y and Z coordinates overlap
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
        if y_overlap > 0 and z_overlap > 0:
            return True
    
    # Check if they share a face along the Y-axis
    if y1_max == y2_min or y1_min == y2_max:
        # Check if the X and Z coordinates overlap
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
        if x_overlap > 0 and z_overlap > 0:
            return True
    
    # Check if they share a face along the Z-axis
    if z1_max == z2_min or z1_min == z2_max:
        # Check if the X and Y coordinates overlap
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        if x_overlap > 0 and y_overlap > 0:
            return True
    
    return False

n = int(input())
cuboids = []

for _ in range(n):
    coords = list(map(int, input().split()))
    x1, y1, z1, x2, y2, z2 = coords
    cuboids.append((x1, y1, z1, x2, y2, z2))

for i in range(n):
    count = 0
    for j in range(n):
        if i != j and share_face(cuboids[i], cuboids[j]):
            count += 1
    print(count)