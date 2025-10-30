# YOUR CODE HERE
def solve():
    N, K = map(int, input().split())
    P = list(map(int, input().split()))
    
    # Convert to 0-indexed
    P = [p - 1 for p in P]
    
    result = [0] * N
    visited = [False] * N
    
    for start in range(N):
        if visited[start]:
            continue
            
        # Find the cycle starting from this position
        cycle = []
        current = start
        
        while not visited[current]:
            visited[current] = True
            cycle.append(current)
            current = P[current]
        
        # Now we have a complete cycle
        cycle_length = len(cycle)
        
        # For each position in the cycle, calculate where it ends up after K operations
        for i, pos in enumerate(cycle):
            # After K operations, position at index i in cycle goes to index (i + K) % cycle_length
            final_index = (i + K) % cycle_length
            result[pos] = cycle[final_index]
    
    # Convert back to 1-indexed and print
    result = [r + 1 for r in result]
    print(*result)

solve()