# YOUR CODE HERE
def solve():
    N, K = map(int, input().split())
    P = list(map(int, input().split()))
    
    # Convert to 0-indexed
    P = [p - 1 for p in P]
    
    # Find where each element goes after K operations
    result = [0] * N
    
    for start in range(N):
        # Find the cycle containing this element
        cycle = []
        current = start
        visited = set()
        
        while current not in visited:
            visited.add(current)
            cycle.append(current)
            current = P[current]
        
        # Find position in cycle after K steps
        cycle_length = len(cycle)
        final_pos = K % cycle_length
        
        # Place the element at its final position
        result[start] = cycle[final_pos] + 1
    
    print(' '.join(map(str, result)))

solve()