# YOUR CODE HERE
def dist(A, B):
    x1, y1 = A
    x2, y2 = B
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if (dx + dy) % 2 != 0:
        return 0
    return max(dx, dy) // 2

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
points = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N)]

total_distance = 0

for i in range(N-1):
    for j in range(i+1, N):
        total_distance += dist(points[i], points[j])

print(total_distance)