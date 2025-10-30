def solve():
    N, M = map(int, input().split())
    constraints = []
    for _ in range(M):
        x, y, z = map(int, input().split())
        constraints.append((x-1, y-1, z))  # Convert to 0-indexed
    
    # Find the maximum bit position needed
    max_z = max((z for _, _, z in constraints), default=0)
    max_bits = max_z.bit_length()
    
    result = [0] * N
    
    for bit in range(max_bits):
        # Build graph for this bit position
        graph = [[] for _ in range(N)]
        for x, y, z in constraints:
            bit_val = (z >> bit) & 1
            graph[x].append((y, bit_val))
            graph[y].append((x, bit_val))
        
        # Check 2-colorability and find optimal coloring
        color = [-1] * N
        
        for start in range(N):
            if color[start] != -1:
                continue
            
            # BFS to color the component
            component = []
            queue = [start]
            color[start] = 0
            component.append(start)
            valid = True
            
            while queue and valid:
                u = queue.pop(0)
                for v, edge_type in graph[u]:
                    if color[v] == -1:
                        color[v] = color[u] ^ edge_type
                        queue.append(v)
                        component.append(v)
                    else:
                        # Check consistency
                        if color[v] != (color[u] ^ edge_type):
                            valid = False
                            break
            
            if not valid:
                print(-1)
                return
            
            # Choose the better coloring for this component
            count_0 = sum(1 for v in component if color[v] == 0)
            count_1 = len(component) - count_0
            
            if count_1 > count_0:
                # Flip all colors in this component
                for v in component:
                    color[v] = 1 - color[v]
        
        # Add contribution from this bit
        for i in range(N):
            if color[i] == 1:
                result[i] += (1 << bit)
    
    print(' '.join(map(str, result)))

solve()