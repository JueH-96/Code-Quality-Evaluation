def solve():
    # Read input
    N, M = map(int, input().split())
    X = []
    Y = []
    Z = []
    for _ in range(M):
        x, y, z = map(int, input().split())
        X.append(x-1)  # Convert to 0-based indexing
        Y.append(y-1)
        Z.append(z)
    
    # Create graph of XOR equations
    graph = [[] for _ in range(N)]
    for i in range(M):
        graph[X[i]].append((Y[i], Z[i]))
        graph[Y[i]].append((X[i], Z[i]))
    
    # Try to solve starting with each component
    def try_solve(start):
        values = [-1] * N
        values[start] = 0  # Try starting with 0
        
        # BFS to propagate values
        queue = [start]
        visited = {start}
        
        while queue:
            curr = queue.pop(0)
            curr_val = values[curr]
            
            # Check all neighbors
            for next_node, xor_val in graph[curr]:
                expected_val = curr_val ^ xor_val
                
                if values[next_node] == -1:
                    # Not visited yet
                    values[next_node] = expected_val
                    queue.append(next_node)
                    visited.add(next_node)
                elif values[next_node] != expected_val:
                    # Contradiction found
                    return None
        
        # Fill remaining unvisited nodes with 0
        for i in range(N):
            if values[i] == -1:
                values[i] = 0
                
        # Verify solution
        for i in range(M):
            if (values[X[i]] ^ values[Y[i]]) != Z[i]:
                return None
        
        return values

    # Try each component as starting point
    min_sum = float('inf')
    best_solution = None
    
    for start in range(N):
        solution = try_solve(start)
        if solution is not None:
            curr_sum = sum(solution)
            if curr_sum < min_sum:
                min_sum = curr_sum
                best_solution = solution
    
    if best_solution is None:
        print(-1)
    else:
        print(*best_solution)

solve()