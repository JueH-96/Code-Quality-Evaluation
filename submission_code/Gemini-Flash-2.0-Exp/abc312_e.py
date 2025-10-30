def solve():
    n = int(input())
    cuboids = []
    for _ in range(n):
        x1, y1, z1, x2, y2, z2 = map(int, input().split())
        cuboids.append(((x1, y1, z1), (x2, y2, z2)))

    def intersects_face(c1, c2):
        x11, y11, z11 = c1[0]
        x12, y12, z12 = c1[1]
        x21, y21, z21 = c2[0]
        x22, y22, z22 = c2[1]

        # Check if they intersect on x faces
        if x11 == x22:
            y_overlap = max(0, min(y12, y22) - max(y11, y21))
            z_overlap = max(0, min(z12, z22) - max(z11, z21))
            if y_overlap > 0 and z_overlap > 0:
                return True
        if x12 == x21:
            y_overlap = max(0, min(y12, y22) - max(y11, y21))
            z_overlap = max(0, min(z12, z22) - max(z11, z21))
            if y_overlap > 0 and z_overlap > 0:
                return True

        # Check if they intersect on y faces
        if y11 == y22:
            x_overlap = max(0, min(x12, x22) - max(x11, x21))
            z_overlap = max(0, min(z12, z22) - max(z11, z21))
            if x_overlap > 0 and z_overlap > 0:
                return True
        if y12 == y21:
            x_overlap = max(0, min(x12, x22) - max(x11, x21))
            z_overlap = max(0, min(z12, z22) - max(z11, z21))
            if x_overlap > 0 and z_overlap > 0:
                return True
        
        # Check if they intersect on z faces
        if z11 == z22:
            x_overlap = max(0, min(x12, x22) - max(x11, x21))
            y_overlap = max(0, min(y12, y22) - max(y11, y21))
            if x_overlap > 0 and y_overlap > 0:
                return True
        if z12 == z21:
            x_overlap = max(0, min(x12, x22) - max(x11, x21))
            y_overlap = max(0, min(y12, y22) - max(y11, y21))
            if x_overlap > 0 and y_overlap > 0:
                return True
        
        return False

    for i in range(n):
        count = 0
        for j in range(n):
            if i != j:
                if intersects_face(cuboids[i], cuboids[j]):
                    count += 1
        print(count)

solve()