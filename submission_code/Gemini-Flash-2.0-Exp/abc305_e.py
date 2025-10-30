from collections import deque

def solve():
    n, m, k = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    guards = []
    for _ in range(k):
        p, h = map(int, input().split())
        guards.append((p, h))
    
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    guarded = [False] * (n + 1)
    
    for p, h in guards:
        q = deque([(p, 0)])
        visited = {p}
        
        while q:
            curr, dist = q.popleft()
            if dist <= h:
                guarded[curr] = True
                
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))
    
    guarded_vertices = [i for i in range(1, n + 1) if guarded[i]]
    
    print(len(guarded_vertices))
    print(*guarded_vertices)

solve()