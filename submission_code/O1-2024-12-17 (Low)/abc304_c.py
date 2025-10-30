import math
import sys
from collections import deque

def main():
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    D = int(input_data[1])

    points = []
    idx = 2
    for _ in range(N):
        x = int(input_data[idx])
        y = int(input_data[idx+1])
        points.append((x, y))
        idx += 2

    # Adjacency list (or we can do on-the-fly checking in BFS)
    visited = [False] * N
    visited[0] = True  # Person 1 is infected initially

    queue = deque([0])
    while queue:
        current = queue.popleft()
        for i in range(N):
            if not visited[i]:
                dx = points[current][0] - points[i][0]
                dy = points[current][1] - points[i][1]
                dist_sq = dx*dx + dy*dy
                if dist_sq <= D*D:
                    visited[i] = True
                    queue.append(i)

    # Output results
    for i in range(N):
        print("Yes" if visited[i] else "No")


# Do not forget to call main!
main()