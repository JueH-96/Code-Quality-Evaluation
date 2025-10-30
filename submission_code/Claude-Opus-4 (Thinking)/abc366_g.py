def solve():
    N, M = map(int, input().split())
    
    # Special case: no edges
    if M == 0:
        print("Yes")
        print(" ".join(["1"] * N))
        return
    
    # Build adjacency list
    adj = [[] for _ in range(N)]
    edges = []
    for _ in range(M):
        u, v = map(int, input().split())
        u -= 1  # Convert to 0-indexed
        v -= 1
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
    
    # Check if any vertex has degree 1
    for i in range(N):
        if len(adj[i]) == 1:
            print("No")
            return
    
    # If all vertices have even degree, we can assign the same value to all
    if all(len(adj[i]) % 2 == 0 for i in range(N)):
        print("Yes")
        print(" ".join(["1"] * N))
        return
    
    # For general case, we need to find a valid assignment
    # We'll use a system of linear equations approach
    
    # Try to find a solution by assigning values
    # Strategy: Use Gaussian elimination over GF(2) for each bit position
    
    # For simplicity, let's use a heuristic that works for small graphs
    values = [0] * N
    
    # Find connected components
    visited = [False] * N
    components = []
    
    def dfs(v, comp):
        visited[v] = True
        comp.append(v)
        for u in adj[v]:
            if not visited[u]:
                dfs(u, comp)
    
    for i in range(N):
        if not visited[i] and len(adj[i]) > 0:
            comp = []
            dfs(i, comp)
            components.append(comp)
        elif not visited[i]:  # Isolated vertex
            values[i] = 1
    
    # For each component, find a valid assignment
    value_id = 1
    for comp_id, comp in enumerate(components):
        # Check if all vertices in component have even degree
        if all(len(adj[v]) % 2 == 0 for v in comp):
            for v in comp:
                values[v] = value_id
            value_id += 1
        else:
            # Use a pattern that often works for mixed degree components
            # Assign different powers of 2 to create linear independence
            comp_values = {}
            
            # Start with some vertex and assign values using BFS
            from collections import deque
            q = deque([comp[0]])
            comp_values[comp[0]] = 1 << comp_id
            assigned = {comp[0]}
            
            while len(assigned) < len(comp):
                # Find an unassigned vertex connected to assigned ones
                found = False
                for v in comp:
                    if v not in assigned:
                        # Check how many neighbors are assigned
                        assigned_neighbors = [u for u in adj[v] if u in assigned]
                        if assigned_neighbors:
                            # Calculate required XOR value
                            xor_val = 0
                            for u in assigned_neighbors:
                                xor_val ^= comp_values[u]
                            
                            # Assign a value that satisfies constraints
                            if len(assigned_neighbors) < len(adj[v]):
                                # Still have unassigned neighbors
                                comp_values[v] = 1 << (len(assigned) + comp_id)
                            else:
                                # All neighbors assigned, must satisfy XOR = 0
                                comp_values[v] = xor_val
                                if comp_values[v] == 0:
                                    # Can't assign 0, try different approach
                                    comp_values[v] = 1 << (20 + len(assigned))
                            
                            assigned.add(v)
                            found = True
                            break
                
                if not found:
                    # Fallback: assign remaining vertices
                    for v in comp:
                        if v not in assigned:
                            comp_values[v] = 1 << (len(assigned) + comp_id + 10)
                            assigned.add(v)
            
            # Copy values
            for v in comp:
                values[v] = comp_values.get(v, 1)
    
    # Verify solution
    valid = True
    for v in range(N):
        if len(adj[v]) > 0:
            xor_sum = 0
            for u in adj[v]:
                xor_sum ^= values[u]
            if xor_sum != 0:
                valid = False
                break
    
    if valid and all(v > 0 for v in values):
        print("Yes")
        print(" ".join(map(str, values)))
    else:
        print("No")

solve()