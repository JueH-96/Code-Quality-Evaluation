# YOUR CODE HERE
def solve():
    N, K = map(int, input().split())
    X = list(map(int, input().split()))
    A = list(map(int, input().split()))
    
    # Convert X to 0-indexed
    X = [x - 1 for x in X]
    
    if K == 0:
        print(*A)
        return
    
    # For each position, find where it goes after K operations
    result = [0] * N
    
    for i in range(N):
        # Find the cycle that position i belongs to
        visited = []
        pos = i
        
        # Trace the cycle
        while pos not in visited:
            visited.append(pos)
            pos = X[pos]
        
        # Find where in the cycle we start
        cycle_start = visited.index(pos)
        cycle = visited[cycle_start:]
        cycle_length = len(cycle)
        
        if i in cycle:
            # Position i is in the cycle
            idx_in_cycle = cycle.index(i)
            final_pos = cycle[(idx_in_cycle + K) % cycle_length]
        else:
            # Position i leads to the cycle but isn't in it
            steps_to_cycle = cycle_start
            if K < steps_to_cycle:
                # We don't reach the cycle
                final_pos = visited[K]
            else:
                # We reach the cycle
                remaining_steps = K - steps_to_cycle
                final_pos = cycle[remaining_steps % cycle_length]
        
        result[i] = A[final_pos]
    
    print(*result)

solve()