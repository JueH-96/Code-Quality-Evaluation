import sys

def solve():
    N = int(sys.stdin.readline())
    P_list = list(map(int, sys.stdin.readline().split()))
    
    # Use 1-indexed P array for easier correspondence with problem statement
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P_list[i]

    # Check if sorted (0 operations)
    is_sorted = True
    for i in range(1, N + 1):
        if P[i] != i:
            is_sorted = False
            break
    if is_sorted:
        sys.stdout.write("0
")
        return

    # Precompute PrefMax and SuffMin for 1-operation check
    # PrefMax[i] = max(P[1]...P[i])
    # SuffMin[i] = min(P[i]...P[N])
    
    PrefMax = [0] * (N + 1)
    # N >= 3, so P[1] is safe to access
    PrefMax[1] = P[1]
    for i in range(2, N + 1):
        PrefMax[i] = max(PrefMax[i-1], P[i])

    # SuffMin[N+1] is a sentinel. 
    # If k=N, k+1 = N+1. The condition for the right part is SuffMin[k+1] == k+1.
    # So, SuffMin[N+1] must be N+1 for this to be true if k=N (suffix is empty).
    SuffMin = [0] * (N + 2)
    SuffMin[N] = P[N] 
    SuffMin[N+1] = N + 1 
    
    for i in range(N - 1, 0, -1): # Iterate from N-1 down to 1
        SuffMin[i] = min(SuffMin[i+1], P[i])

    # Check for 1-operation solution
    found_one_op_solution = False
    for k_val in range(1, N + 1): # k_val is the k chosen for the operation
        if P[k_val] == k_val: # P_k = k condition
            
            cond_L = False # True if left part P[1...k_val-1] can become 1...k_val-1
            if k_val == 1: # Prefix P[1...k_val-1] is empty
                cond_L = True
            elif PrefMax[k_val-1] == k_val-1: # max(P[1...k_val-1]) == k_val-1
                cond_L = True
            
            cond_R = False # True if right part P[k_val+1...N] can become k_val+1...N
            if k_val == N: # Suffix P[k_val+1...N] is empty
                cond_R = True
            elif SuffMin[k_val+1] == k_val+1: # min(P[k_val+1...N]) == k_val+1
                cond_R = True
            
            if cond_L and cond_R:
                found_one_op_solution = True
                break
    
    if found_one_op_solution:
        sys.stdout.write("1
")
        return

    # If not 0-op and not 1-op:
    # Check for 3 operations: P[1]=N AND P[N]=1
    if P[1] == N and P[N] == 1:
        sys.stdout.write("3
")
        return
        
    # Otherwise, it must be 2 operations.
    sys.stdout.write("2
")

# Main loop for T test cases
T = int(sys.stdin.readline())
for _ in range(T):
    solve()