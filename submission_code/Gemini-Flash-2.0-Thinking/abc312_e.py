def solve():
    n = int(input())
    cuboids_data = []
    for _ in range(n):
        x1, y1, z1, x2, y2, z2 = map(int, input().split())
        cuboids_data.append(tuple(sorted([(x1, y1, z1), (x2, y2, z2)])))

    cuboids = []
    for (x1, y1, z1), (x2, y2, z2) in cuboids_data:
        cuboids.append((min(x1, x2), min(y1, y2), min(z1, z2), max(x1, x2), max(y1, y2), max(z1, z2)))

    def share_face(c1_idx, c2_idx):
        xmin1, ymin1, zmin1, xmax1, ymax1, zmax1 = cuboids[c1_idx]
        xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = cuboids[c2_idx]

        # Share XY face
        if zmax1 == zmin2 and max(xmin1, xmin2) < min(xmax1, xmax2) and max(ymin1, ymin2) < min(ymax1, ymax2):
            return True
        if zmin1 == zmax2 and max(xmin1, xmin2) < min(xmax1, xmax2) and max(ymin1, ymin2) < min(ymax1, ymax2):
            return True

        # Share XZ face
        if ymax1 == ymin2 and max(xmin1, xmin2) < min(xmax1, xmax2) and max(zmin1, zmin2) < min(zmax1, zmax2):
            return True
        if ymin1 == ymax2 and max(xmin1, xmin2) < min(xmax1, xmax2) and max(zmin1, zmin2) < min(zmax1, zmax2):
            return True

        # Share YZ face
        if xmax1 == xmin2 and max(ymin1, ymin2) < min(ymax1, ymax2) and max(zmin1, zmin2) < min(zmax1, zmax2):
            return True
        if xmin1 == xmax2 and max(ymin1, ymin2) < min(ymax1, ymax2) and max(zmin1, zmin2) < min(zmax1, zmax2):
            return True

        return False

    face_counts = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if share_face(i, j):
                face_counts[i] += 1
                face_counts[j] += 1

    for count in face_counts:
        print(count)

solve()