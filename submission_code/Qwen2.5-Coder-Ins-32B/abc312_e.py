# YOUR CODE HERE
from collections import defaultdict

def count_shared_faces(cuboids):
    x_faces = defaultdict(list)
    y_faces = defaultdict(list)
    z_faces = defaultdict(list)
    
    for i, (x1, y1, z1, x2, y2, z2) in enumerate(cuboids):
        x_faces[(x1, y1, y2, z1, z2)].append(i)
        x_faces[(x2, y1, y2, z1, z2)].append(i)
        y_faces[(y1, x1, x2, z1, z2)].append(i)
        y_faces[(y2, x1, x2, z1, z2)].append(i)
        z_faces[(z1, x1, x2, y1, y2)].append(i)
        z_faces[(z2, x1, x2, y1, y2)].append(i)
    
    shared_faces_count = [0] * len(cuboids)
    
    for face in x_faces.values():
        for i in range(len(face)):
            for j in range(i + 1, len(face)):
                shared_faces_count[face[i]] += 1
                shared_faces_count[face[j]] += 1
    
    for face in y_faces.values():
        for i in range(len(face)):
            for j in range(i + 1, len(face)):
                shared_faces_count[face[i]] += 1
                shared_faces_count[face[j]] += 1
    
    for face in z_faces.values():
        for i in range(len(face)):
            for j in range(i + 1, len(face)):
                shared_faces_count[face[i]] += 1
                shared_faces_count[face[j]] += 1
    
    return shared_faces_count

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
cuboids = []

index = 1
for _ in range(N):
    x1 = int(data[index])
    y1 = int(data[index + 1])
    z1 = int(data[index + 2])
    x2 = int(data[index + 3])
    y2 = int(data[index + 4])
    z2 = int(data[index + 5])
    cuboids.append((x1, y1, z1, x2, y2, z2))
    index += 6

result = count_shared_faces(cuboids)
print("
".join(map(str, result)))