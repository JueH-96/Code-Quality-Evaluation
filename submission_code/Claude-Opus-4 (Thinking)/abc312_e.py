def ranges_overlap(a1, a2, b1, b2):
    return max(a1, b1) < min(a2, b2)

def cuboids_share_face(c1, c2):
    x1_1, y1_1, z1_1, x1_2, y1_2, z1_2 = c1
    x2_1, y2_1, z2_1, x2_2, y2_2, z2_2 = c2
    
    # Check if they share a face along X-axis
    if (x1_2 == x2_1 or x1_1 == x2_2) and \
       ranges_overlap(y1_1, y1_2, y2_1, y2_2) and \
       ranges_overlap(z1_1, z1_2, z2_1, z2_2):
        return True
    
    # Check if they share a face along Y-axis
    if (y1_2 == y2_1 or y1_1 == y2_2) and \
       ranges_overlap(x1_1, x1_2, x2_1, x2_2) and \
       ranges_overlap(z1_1, z1_2, z2_1, z2_2):
        return True
    
    # Check if they share a face along Z-axis
    if (z1_2 == z2_1 or z1_1 == z2_2) and \
       ranges_overlap(x1_1, x1_2, x2_1, x2_2) and \
       ranges_overlap(y1_1, y1_2, y2_1, y2_2):
        return True
    
    return False

n = int(input())
cuboids = []
for _ in range(n):
    coords = list(map(int, input().split()))
    cuboids.append(coords)

for i in range(n):
    count = 0
    for j in range(n):
        if i != j and cuboids_share_face(cuboids[i], cuboids[j]):
            count += 1
    print(count)