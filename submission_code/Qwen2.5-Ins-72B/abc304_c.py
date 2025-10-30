import math
from collections import deque

# Read input
N, D = map(int, input().split())
coordinates = [tuple(map(int, input().split())) for _ in range(N)]

# Function to calculate Euclidean distance
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Initialize infection status
infected = [False] * N
infected[0] = True

# BFS to spread the infection
queue = deque([0])
while queue:
    current = queue.popleft()
    for i in range(N):
        if not infected[i] and distance(coordinates[current], coordinates[i]) <= D:
            infected[i] = True
            queue.append(i)

# Output the result
for i in range(N):
    print("Yes" if infected[i] else "No")