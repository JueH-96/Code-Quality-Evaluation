import sys
import math
from collections import deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
D = int(data[1])

points = []
for i in range(N):
    x = int(data[2 * i + 2])
    y = int(data[2 * i + 3])
    points.append((x, y))

# Initialize the infected status of each person
infected = [False] * N
infected[0] = True

# Use a queue to perform BFS
queue = deque([0])

while queue:
    current = queue.popleft()
    for i in range(N):
        if not infected[i]:
            dist = math.sqrt((points[current][0] - points[i][0]) ** 2 + (points[current][1] - points[i][1]) ** 2)
            if dist <= D:
                infected[i] = True
                queue.append(i)

# Output the results
for i in range(N):
    print("Yes" if infected[i] else "No")