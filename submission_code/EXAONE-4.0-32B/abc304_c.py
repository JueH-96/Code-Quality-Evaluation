import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    d = int(data[1])
    points = []
    index = 2
    for i in range(n):
        x = int(data[index])
        y = int(data[index + 1])
        index += 2
        points.append((x, y))
    
    d_sq = d * d
    visited = [False] * n
    q = deque()
    visited[0] = True
    q.append(0)
    
    while q:
        u = q.popleft()
        for v in range(n):
            if not visited[v]:
                dx = points[u][0] - points[v][0]
                dy = points[u][1] - points[v][1]
                dist_sq = dx * dx + dy * dy
                if dist_sq <= d_sq:
                    visited[v] = True
                    q.append(v)
    
    for i in range(n):
        print("Yes" if visited[i] else "No")

if __name__ == "__main__":
    main()