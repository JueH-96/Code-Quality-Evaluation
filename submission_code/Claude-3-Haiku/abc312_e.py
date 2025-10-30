from collections import defaultdict

def count_shared_faces(N, cuboids):
    # Create a dictionary to store the coordinates of each cuboid
    coords = defaultdict(list)
    for i, (x1, y1, z1, x2, y2, z2) in enumerate(cuboids):
        coords[(x1, y1, z1, x2, y2, z2)].append(i)

    # Count the number of shared faces for each cuboid
    result = [0] * N
    for (x1, y1, z1, x2, y2, z2), indices in coords.items():
        for i in indices:
            for j in indices:
                if i != j:
                    result[i] += 1
    return result

# Read the input
N = int(input())
cuboids = []
for _ in range(N):
    x1, y1, z1, x2, y2, z2 = map(int, input().split())
    cuboids.append((x1, y1, z1, x2, y2, z2))

# Solve the problem and print the result
result = count_shared_faces(N, cuboids)
for count in result:
    print(count)