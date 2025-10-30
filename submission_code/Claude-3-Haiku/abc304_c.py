import math

# Read the input
N, D = map(int, input().split())
coordinates = []
for _ in range(N):
    x, y = map(int, input().split())
    coordinates.append((x, y))

# Determine if each person is infected
for i in range(N):
    infected = False
    for j in range(N):
        if i != j:
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distance <= D:
                infected = True
                break
    if infected:
        print("Yes")
    else:
        print("No")