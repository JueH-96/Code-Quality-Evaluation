def solve():
    # Read input
    N, K = map(int, input().split())
    P = list(map(int, input().split()))
    
    # Convert to 0-based indexing for easier handling
    P = [x-1 for x in P]
    
    # Find cycle length for each position
    visited = [False] * N
    cycles = []
    
    # Find all cycles in the permutation
    for i in range(N):
        if not visited[i]:
            cycle = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                curr = P[curr]
            if cycle:
                cycles.append(cycle)
    
    # For each position, find where it will end up after K operations
    result = [0] * N
    
    # Process each cycle
    for cycle in cycles:
        cycle_len = len(cycle)
        # For each position in this cycle
        for i, pos in enumerate(cycle):
            # Calculate final position after K moves
            new_pos = cycle[(i + K) % cycle_len]
            result[pos] = P[new_pos]
    
    # Convert back to 1-based indexing and print
    print(*[x+1 for x in result])

# Run the solution
solve()