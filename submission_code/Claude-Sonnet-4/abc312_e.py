def cuboids_share_face(c1, c2):
    # c1 and c2 are tuples (x1, y1, z1, x2, y2, z2)
    x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = c1
    x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = c2
    
    # Check if they share a face along x-axis
    if (x1_max == x2_min or x1_min == x2_max):
        # Check if they overlap in y and z
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
        if y_overlap > 0 and z_overlap > 0:
            return True
    
    # Check if they share a face along y-axis
    if (y1_max == y2_min or y1_min == y2_max):
        # Check if they overlap in x and z
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
        if x_overlap > 0 and z_overlap > 0:
            return True
    
    # Check if they share a face along z-axis
    if (z1_max == z2_min or z1_min == z2_max):
        # Check if they overlap in x and y
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        if x_overlap > 0 and y_overlap > 0:
            return True
    
    return False

# Read input
n = int(input())
cuboids = []

for _ in range(n):
    coords = list(map(int, input().split()))
    x1, y1, z1, x2, y2, z2 = coords
    cuboids.append((x1, y1, z1, x2, y2, z2))

# For each cuboid, count how many others share a face with it
results = []
for i in range(n):
    count = 0
    for j in range(n):
        if i != j and cuboids_share_face(cuboids[i], cuboids[j]):
            count += 1
    results.append(count)

# Output results
for result in results:
    print(result)