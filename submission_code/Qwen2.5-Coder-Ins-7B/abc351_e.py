# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
points = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N)]

def dist(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

total_dist = 0
for i in range(N-1):
    for j in range(i+1, N):
        total_dist += dist(points[i], points[j])

print(total_dist)