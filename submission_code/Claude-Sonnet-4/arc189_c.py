def solve():
    N, X = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    P = list(map(int, input().split()))
    Q = list(map(int, input().split()))
    
    # Convert to 0-indexed
    X -= 1
    P = [p - 1 for p in P]
    Q = [q - 1 for q in Q]
    
    # Check if already solved
    if all(A[i] == 0 and B[i] == 0 for i in range(N) if i != X):
        return 0
    
    # Find minimum steps to reach X for each position
    def min_steps_to_reach(perm, target):
        steps = [float('inf')] * N
        steps[target] = 0
        
        # Work backwards from target
        changed = True
        while changed:
            changed = False
            for i in range(N):
                if steps[perm[i]] > steps[i] + 1:
                    steps[perm[i]] = steps[i] + 1
                    changed = True
        
        return steps
    
    red_steps = min_steps_to_reach(P, X)
    blue_steps = min_steps_to_reach(Q, X)
    
    # Check feasibility and find minimum operations
    max_steps = 0
    
    for i in range(N):
        if A[i] > 0:
            if red_steps[i] == float('inf'):
                return -1
            max_steps = max(max_steps, red_steps[i])
        
        if B[i] > 0:
            if blue_steps[i] == float('inf'):
                return -1
            max_steps = max(max_steps, blue_steps[i])
    
    return max_steps

print(solve())