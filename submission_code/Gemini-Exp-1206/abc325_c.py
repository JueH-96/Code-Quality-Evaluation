def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    sensors = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '#':
                sensors.append((r, c))
    
    if not sensors:
        print(0)
        return
    
    adj = [[] for _ in range(len(sensors))]
    for i in range(len(sensors)):
        for j in range(i + 1, len(sensors)):
            r1, c1 = sensors[i]
            r2, c2 = sensors[j]
            if max(abs(r1 - r2), abs(c1 - c2)) == 1:
                adj[i].append(j)
                adj[j].append(i)
    
    visited = [False] * len(sensors)
    count = 0
    
    def dfs(u):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                dfs(v)
    
    for i in range(len(sensors)):
        if not visited[i]:
            dfs(i)
            count += 1
            
    print(count)

solve()