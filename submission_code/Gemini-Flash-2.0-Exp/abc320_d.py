def solve():
    n, m = map(int, input().split())
    
    adj = [[] for _ in range(n + 1)]
    
    for _ in range(m):
        a, b, x, y = map(int, input().split())
        adj[a].append((b, x, y))
        
    coords = {}
    coords[1] = (0, 0)
    
    q = [1]
    
    while q:
        curr = q.pop(0)
        
        for neighbor, dx, dy in adj[curr]:
            if neighbor not in coords:
                coords[neighbor] = (coords[curr][0] + dx, coords[curr][1] + dy)
                q.append(neighbor)
    
    for i in range(1, n + 1):
        if i in coords:
            print(coords[i][0], coords[i][1])
        else:
            print("undecidable")

solve()