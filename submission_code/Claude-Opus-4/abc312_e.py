def check_interval_overlap(a1, a2, b1, b2):
    """Check if two intervals [a1, a2] and [b1, b2] overlap"""
    return max(a1, b1) < min(a2, b2)

def check_interval_touch(a1, a2, b1, b2):
    """Check if two intervals touch (one ends where the other begins)"""
    return a2 == b1 or b2 == a1

def share_face(cuboid1, cuboid2):
    """Check if two cuboids share a face"""
    x1_1, y1_1, z1_1, x1_2, y1_2, z1_2 = cuboid1
    x2_1, y2_1, z2_1, x2_2, y2_2, z2_2 = cuboid2
    
    # Check if they share a face parallel to YZ plane (touching in X dimension)
    if check_interval_touch(x1_1, x1_2, x2_1, x2_2):
        if check_interval_overlap(y1_1, y1_2, y2_1, y2_2) and check_interval_overlap(z1_1, z1_2, z2_1, z2_2):
            return True
    
    # Check if they share a face parallel to XZ plane (touching in Y dimension)
    if check_interval_touch(y1_1, y1_2, y2_1, y2_2):
        if check_interval_overlap(x1_1, x1_2, x2_1, x2_2) and check_interval_overlap(z1_1, z1_2, z2_1, z2_2):
            return True
    
    # Check if they share a face parallel to XY plane (touching in Z dimension)
    if check_interval_touch(z1_1, z1_2, z2_1, z2_2):
        if check_interval_overlap(x1_1, x1_2, x2_1, x2_2) and check_interval_overlap(y1_1, y1_2, y2_1, y2_2):
            return True
    
    return False

# Read input
n = int(input())
cuboids = []
for _ in range(n):
    coords = list(map(int, input().split()))
    cuboids.append(coords)

# For each cuboid, count how many others share a face with it
for i in range(n):
    count = 0
    for j in range(n):
        if i != j and share_face(cuboids[i], cuboids[j]):
            count += 1
    print(count)