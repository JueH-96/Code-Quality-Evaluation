def bellman_ford(n, edges, start):
    dist = [float('inf')] * (n+1)
    dist[start] = 0
    
    for _ in range(n-1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                
    return dist

def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))
    
    # Try all possible starting vertices
    min_cost = float('inf')
    
    # For each starting vertex
    for start in range(1, n+1):
        # For each subset of vertices that must be visited
        for state in range(1 << n):
            if not (state & (1 << (start-1))):
                continue
                
            # dp[v][s] = minimum cost to reach vertex v having visited vertices in state s
            dp = {}
            for v in range(1, n+1):
                for s in range(1 << n):
                    dp[(v,s)] = float('inf')
            
            dp[(start, 1 << (start-1))] = 0
            
            # For each state
            for s in range(1 << n):
                # For each vertex we're currently at
                for v in range(1, n+1):
                    if dp[(v,s)] == float('inf'):
                        continue
                    # Try extending path using each edge
                    for u, nv, w in edges:
                        if u != v:
                            continue
                        new_state = s | (1 << (nv-1))
                        dp[(nv,new_state)] = min(dp[(nv,new_state)], dp[(v,s)] + w)
            
            # Check if all required vertices were visited
            required = (1 << n) - 1
            if (state & required) != required:
                continue
                
            # Find minimum among all ending vertices
            for v in range(1, n+1):
                min_cost = min(min_cost, dp[(v,state)])
    
    if min_cost == float('inf'):
        print("No")
    else:
        print(min_cost)

solve()