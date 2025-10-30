import sys
from collections import defaultdict, deque

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N, M = int(data[0]), int(data[1])
    idx = 2
    edges = defaultdict(list)
    for _ in range(M):
        A, B, X, Y = int(data[idx]), int(data[idx+1]), int(data[idx+2]), int(data[idx+3])
        edges[A].append((B, X, Y))
        edges[B].append((A, -X, -Y))
        idx += 4
    
    coords = [(None, None)] * (N + 1)
    coords[1] = (0, 0)
    visited = [False] * (N + 1)
    visited[1] = True
    queue = deque([1])
    
    while queue:
        node = queue.popleft()
        for neighbor, dx, dy in edges[node]:
            if coords[neighbor] == (None, None):
                coords[neighbor] = (coords[node][0] + dx, coords[node][1] + dy)
                visited[neighbor] = True
                queue.append(neighbor)
    
    for i in range(1, N + 1):
        if not visited[i]:
            print("undecidable")
        else:
            print(coords[i][0], coords[i][1])

solve()