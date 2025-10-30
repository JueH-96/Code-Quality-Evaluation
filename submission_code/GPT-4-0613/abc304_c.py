import sys
import math
from collections import deque

def main():
    N, D = map(int, sys.stdin.readline().split())
    points = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
    graph = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if math.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2) <= D:
                graph[i].append(j)
                graph[j].append(i)
    visited = [False]*N
    queue = deque([0])
    visited[0] = True
    while queue:
        v = queue.popleft()
        for i in graph[v]:
            if not visited[i]:
                visited[i] = True
                queue.append(i)
    for v in visited:
        print('Yes' if v else 'No')

if __name__ == "__main__":
    main()