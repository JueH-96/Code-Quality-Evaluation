import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
cuboids = []
for i in range(N):
    x1, y1, z1, x2, y2, z2 = map(int, data[1 + i * 6: 1 + (i + 1) * 6])
    cuboids.append((x1, y1, z1, x2, y2, z2))

# Create dictionaries to store the faces of the cuboids
x_faces = defaultdict(list)
y_faces = defaultdict(list)
z_faces = defaultdict(list)

for i, (x1, y1, z1, x2, y2, z2) in enumerate(cuboids):
    x_faces[(x1, y1, z1, y2, z2)].append(i)
    x_faces[(x2, y1, z1, y2, z2)].append(i)
    y_faces[(y1, x1, z1, x2, z2)].append(i)
    y_faces[(y2, x1, z1, x2, z2)].append(i)
    z_faces[(z1, x1, y1, x2, y2)].append(i)
    z_faces[(z2, x1, y1, x2, y2)].append(i)

# Count the number of shared faces for each cuboid
shared_faces = [0] * N

for face_dict in [x_faces, y_faces, z_faces]:
    for face, indices in face_dict.items():
        if len(indices) > 1:
            for i in indices:
                shared_faces[i] += len(indices) - 1

# Print the results
for count in shared_faces:
    print(count)