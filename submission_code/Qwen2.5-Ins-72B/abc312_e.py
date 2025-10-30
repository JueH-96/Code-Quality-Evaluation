# YOUR CODE HERE
from collections import defaultdict

def read_input():
    N = int(input())
    cuboids = []
    for _ in range(N):
        x1, y1, z1, x2, y2, z2 = map(int, input().split())
        cuboids.append((x1, y1, z1, x2, y2, z2))
    return N, cuboids

def get_faces(cuboid):
    x1, y1, z1, x2, y2, z2 = cuboid
    faces = [
        ((x1, y1, z1), (x2, y1, z1), (x2, y2, z1), (x1, y2, z1)),  # bottom
        ((x1, y1, z2), (x2, y1, z2), (x2, y2, z2), (x1, y2, z2)),  # top
        ((x1, y1, z1), (x1, y1, z2), (x1, y2, z2), (x1, y2, z1)),  # front
        ((x2, y1, z1), (x2, y1, z2), (x2, y2, z2), (x2, y2, z1)),  # back
        ((x1, y1, z1), (x2, y1, z1), (x2, y1, z2), (x1, y1, z2)),  # left
        ((x1, y2, z1), (x2, y2, z1), (x2, y2, z2), (x1, y2, z2))   # right
    ]
    return faces

def face_intersects(face1, face2):
    (x11, y11, z11), (x12, y12, z12), (x13, y13, z13), (x14, y14, z14) = face1
    (x21, y21, z21), (x22, y22, z22), (x23, y23, z23), (x24, y24, z24) = face2

    x_min1, x_max1 = min(x11, x12, x13, x14), max(x11, x12, x13, x14)
    y_min1, y_max1 = min(y11, y12, y13, y14), max(y11, y12, y13, y14)
    z_min1, z_max1 = min(z11, z12, z13, z14), max(z11, z12, z13, z14)

    x_min2, x_max2 = min(x21, x22, x23, x24), max(x21, x22, x23, x24)
    y_min2, y_max2 = min(y21, y22, y23, y24), max(y21, y22, y23, y24)
    z_min2, z_max2 = min(z21, z22, z23, z24), max(z21, z22, z23, z24)

    return (x_min1 <= x_max2 and x_max1 >= x_min2 and
            y_min1 <= y_max2 and y_max1 >= y_min2 and
            z_min1 <= z_max2 and z_max1 >= z_min2)

def count_shared_faces(N, cuboids):
    face_map = defaultdict(list)
    for i, cuboid in enumerate(cuboids):
        faces = get_faces(cuboid)
        for face in faces:
            face_map[face].append(i)

    result = [0] * N
    for i in range(N):
        faces = get_faces(cuboids[i])
        for face in faces:
            for j in face_map[face]:
                if i != j:
                    result[i] += 1
                    break
    return result

N, cuboids = read_input()
result = count_shared_faces(N, cuboids)
for r in result:
    print(r)