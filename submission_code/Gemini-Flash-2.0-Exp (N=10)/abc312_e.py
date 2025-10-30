def solve():
    n = int(input())
    cuboids = []
    for _ in range(n):
        x1, y1, z1, x2, y2, z2 = map(int, input().split())
        cuboids.append(((x1, y1, z1), (x2, y2, z2)))

    def check_face_overlap(cuboid1, cuboid2):
        (x11, y11, z11), (x12, y12, z12) = cuboid1
        (x21, y21, z21), (x22, y22, z22) = cuboid2

        # Check for overlap on x-faces
        if (x11 == x22 or x12 == x21):
            if max(y11, y21) < min(y12, y22) and max(z11, z21) < min(z12, z22):
                return True
        # Check for overlap on y-faces
        if (y11 == y22 or y12 == y21):
            if max(x11, x21) < min(x12, x22) and max(z11, z21) < min(z12, z22):
                return True
        # Check for overlap on z-faces
        if (z11 == z22 or z12 == z21):
            if max(x11, x21) < min(x12, x22) and max(y11, y21) < min(y12, y22):
                return True
        return False

    for i in range(n):
        count = 0
        for j in range(n):
            if i != j:
                if check_face_overlap(cuboids[i], cuboids[j]):
                    count += 1
        print(count)

solve()