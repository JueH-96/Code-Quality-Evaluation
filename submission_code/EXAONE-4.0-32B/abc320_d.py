import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    
    graph = [[] for _ in range(n+1)]
    
    for _ in range(m):
        a = int(next(it))
        b = int(next(it))
        x = int(next(it))
        y = int(next(it))
        graph[a].append((b, x, y))
        graph[b].append((a, -x, -y))
        
    coords = [None] * (n+1)
    visited = [False] * (n+1)
    queue = deque()
    
    visited[1] = True
    coords[1] = (0, 0)
    queue.append(1)
    
    while queue:
        u = queue.popleft()
        for (v, dx, dy) in graph[u]:
            if not visited[v]:
                visited[v] = True
                cur_x, cur_y = coords[u]
                coords[v] = (cur_x + dx, cur_y + dy)
                queue.append(v)
                
    for i in range(1, n+1):
        if visited[i]:
            print(f"{coords[i][0]} {coords[i][1]}")
        else:
            print("undecidable")

if __name__ == "__main__":
    main()