import sys
from collections import deque

def main():
    n, m = map(int, sys.stdin.readline().split())
    coords = [None] * (n + 1)
    coords[1] = (0, 0)
    
    adj = [[] for _ in range(n + 1)]
    
    for _ in range(m):
        a, b, x, y = map(int, sys.stdin.readline().split())
        adj[a].append((b, x, y))
        adj[b].append((a, -x, -y))
    
    queue = deque()
    queue.append(1)
    
    while queue:
        u = queue.popleft()
        for (v, dx, dy) in adj[u]:
            if coords[v] is None:
                coords[v] = (coords[u][0] + dx, coords[u][1] + dy)
                queue.append(v)
    
    for i in range(1, n + 1):
        if coords[i] is not None:
            print(coords[i][0], coords[i][1])
        else:
            print("undecidable")

if __name__ == '__main__':
    main()