# YOUR CODE HERE
def solve():
    N, K = map(int, input().split())
    X = list(map(int, input().split()))
    A = list(map(int, input().split()))
    
    # Convert X to 0-based indexing
    X = [x - 1 for x in X]
    
    if K == 0:
        print(*A)
        return
    
    # Find cycle by applying transformations until we see a repeated state
    seen = {}
    states = []
    current = A[:]
    
    step = 0
    while True:
        state_key = tuple(current)
        if state_key in seen:
            cycle_start = seen[state_key]
            cycle_length = step - cycle_start
            break
        
        seen[state_key] = step
        states.append(current[:])
        
        # Apply one transformation
        new_current = [0] * N
        for i in range(N):
            new_current[i] = current[X[i]]
        current = new_current
        step += 1
    
    # Now we know the cycle starts at position cycle_start and has length cycle_length
    if K < cycle_start:
        # K is before the cycle starts
        result = states[K]
    else:
        # K is within or after the cycle
        remaining_steps = (K - cycle_start) % cycle_length
        result = states[cycle_start + remaining_steps]
    
    print(*result)

solve()