def has_shared_face(c1, c2):
    # Extract coordinates
    x11, y11, z11, x12, y12, z12 = c1
    x21, y21, z21, x22, y22, z22 = c2
    
    # Check if cuboids share a face
    # For shared face, two coordinates must match exactly and third must overlap
    
    # Check x-face
    if (x11 == x22 or x12 == x21):
        # y ranges must overlap
        y_overlap = not (y11 >= y22 or y21 >= y12)
        # z ranges must overlap
        z_overlap = not (z11 >= z22 or z21 >= z12)
        if y_overlap and z_overlap:
            return True
            
    # Check y-face
    if (y11 == y22 or y12 == y21):
        # x ranges must overlap
        x_overlap = not (x11 >= x22 or x21 >= x12)
        # z ranges must overlap
        z_overlap = not (z11 >= z22 or z21 >= z12)
        if x_overlap and z_overlap:
            return True
            
    # Check z-face
    if (z11 == z22 or z12 == z21):
        # x ranges must overlap
        x_overlap = not (x11 >= x22 or x21 >= x12)
        # y ranges must overlap
        y_overlap = not (y11 >= y22 or y21 >= y12)
        if x_overlap and y_overlap:
            return True
            
    return False

N = int(input())
cuboids = []
for _ in range(N):
    x1, y1, z1, x2, y2, z2 = map(int, input().split())
    cuboids.append((x1, y1, z1, x2, y2, z2))

for i in range(N):
    shared_faces = 0
    for j in range(N):
        if i != j and has_shared_face(cuboids[i], cuboids[j]):
            shared_faces += 1
    print(shared_faces)