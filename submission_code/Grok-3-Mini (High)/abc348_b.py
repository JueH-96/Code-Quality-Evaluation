import sys
data = list(map(int, sys.stdin.read().split()))
N = data[0]
points = [(data[2*i + 1], data[2*i + 2]) for i in range(N)]

for i in range(N):
    max_sq_dist = -1
    best_j = -1
    for j in range(N):
        if j != i:
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            sq_dist = dx*dx + dy*dy
            if sq_dist > max_sq_dist:
                max_sq_dist = sq_dist
                best_j = j
    # Output the ID of the farthest point
    print(best_j + 1)