def intervals_overlap(a1, a2, b1, b2):
    return max(a1, b1) < min(a2, b2)

def cuboids_share_face(cuboid1, cuboid2):
    x1a, y1a, z1a, x2a, y2a, z2a = cuboid1
    x1b, y1b, z1b, x2b, y2b, z2b = cuboid2
    
    # Check if they're adjacent in X dimension and overlap in Y and Z
    if (x2a == x1b or x2b == x1a):
        if intervals_overlap(y1a, y2a, y1b, y2b) and intervals_overlap(z1a, z2a, z1b, z2b):
            return True
    
    # Check if they're adjacent in Y dimension and overlap in X and Z
    if (y2a == y1b or y2b == y1a):
        if intervals_overlap(x1a, x2a, x1b, x2b) and intervals_overlap(z1a, z2a, z1b, z2b):
            return True
    
    # Check if they're adjacent in Z dimension and overlap in X and Y
    if (z2a == z1b or z2b == z1a):
        if intervals_overlap(x1a, x2a, x1b, x2b) and intervals_overlap(y1a, y2a, y1b, y2b):
            return True
    
    return False

n = int(input())
cuboids = []
for _ in range(n):
    coords = list(map(int, input().split()))
    cuboids.append(coords)

result = []
for i in range(n):
    count = 0
    for j in range(n):
        if i != j and cuboids_share_face(cuboids[i], cuboids[j]):
            count += 1
    result.append(count)

for count in result:
    print(count)