def solve():
    N, M = map(int, input().split())
    
    # Build adjacency list with XOR values
    adj = [[] for _ in range(N + 1)]
    
    for _ in range(M):
        x, y, z = map(int, input().split())
        if x == y:
            # Self-loop: A[x] XOR A[x] = 0, so z must be 0
            if z != 0:
                print(-1)
                return
        else:
            adj[x].append((y, z))
            adj[y].append((x, z))
    
    # Find connected components and solve for each
    visited = [False] * (N + 1)
    result = [0] * (N + 1)
    
    for start in range(1, N + 1):
        if visited[start]:
            continue
            
        # BFS to find component
        component = []
        queue = [start]
        visited[start] = True
        
        # Store XOR distance from start node
        xor_dist = [None] * (N + 1)
        xor_dist[start] = 0
        
        valid = True
        
        while queue and valid:
            u = queue.pop(0)
            component.append(u)
            
            for v, xor_val in adj[u]:
                if xor_dist[v] is None:
                    xor_dist[v] = xor_dist[u] ^ xor_val
                    visited[v] = True
                    queue.append(v)
                else:
                    # Check consistency
                    if xor_dist[v] != (xor_dist[u] ^ xor_val):
                        valid = False
                        break
        
        if not valid:
            print(-1)
            return
        
        # Try different values for the start node to minimize sum
        min_sum = float('inf')
        best_start_val = 0
        
        # We need to try values up to some reasonable limit
        # Since XOR preserves bits, we only need to try up to max possible value
        max_xor = 0
        for node in component:
            max_xor = max(max_xor, xor_dist[node])
        
        # Try values from 0 to a reasonable upper bound
        for start_val in range(min(1024, max_xor + 1)):
            current_sum = 0
            for node in component:
                current_sum += start_val ^ xor_dist[node]
            
            if current_sum < min_sum:
                min_sum = current_sum
                best_start_val = start_val
        
        # Assign values to this component
        for node in component:
            result[node] = best_start_val ^ xor_dist[node]
    
    print(*result[1:])

solve()