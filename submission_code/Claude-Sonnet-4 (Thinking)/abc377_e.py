def solve():
    N, K = map(int, input().split())
    P = list(map(int, input().split()))
    
    # Convert to 0-indexed
    P = [p - 1 for p in P]
    
    # Compute P^K using cycle decomposition
    result = [0] * N
    visited = [False] * N
    
    for start in range(N):
        if visited[start]:
            continue
        
        # Find the cycle containing start
        cycle = []
        current = start
        while current not in cycle:
            cycle.append(current)
            current = P[current]
        
        # Mark all elements in the cycle as visited
        for pos in cycle:
            visited[pos] = True
        
        cycle_length = len(cycle)
        effective_k = K % cycle_length
        
        # For each position in the cycle, compute where it goes after K steps
        for i, pos in enumerate(cycle):
            target_index = (i + effective_k) % cycle_length
            target_pos = cycle[target_index]
            result[pos] = target_pos
    
    # Convert back to 1-indexed and print
    result = [r + 1 for r in result]
    print(' '.join(map(str, result)))

solve()