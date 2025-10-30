import math
from collections import deque

# Read input
N, D = map(int, input().split())
coordinates = []
for _ in range(N):
    x, y = map(int, input().split())
    coordinates.append((x, y))

# Function to calculate Euclidean distance
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# BFS to find all infected people
infected = [False] * N
infected[0] = True  # Person 1 (index 0) is initially infected
queue = deque([0])

while queue:
    current = queue.popleft()
    current_pos = coordinates[current]
    
    # Check all other people
    for i in range(N):
        if not infected[i]:  # If person i is not infected yet
            if distance(current_pos, coordinates[i]) <= D:
                infected[i] = True
                queue.append(i)

# Output results
for i in range(N):
    if infected[i]:
        print("Yes")
    else:
        print("No")