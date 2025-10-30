import sys

def solve():
    N = int(sys.stdin.readline())
    S_str = sys.stdin.readline().strip()
    C_list = list(map(int, sys.stdin.readline().split()))

    S_chars = [int(c) for c in S_str] 

    cost_to_0 = [0] * N
    cost_to_1 = [0] * N
    for k in range(N):
        if S_chars[k] == 0:
            cost_to_0[k] = 0
            cost_to_1[k] = C_list[k]
        else: 
            cost_to_0[k] = C_list[k]
            cost_to_1[k] = 0
    
    val_L0_terms = [(cost_to_0[k] if k % 2 == 1 else 0) for k in range(N)]
    val_E1_terms = [(cost_to_1[k] if k % 2 == 0 else 0) for k in range(N)]
    val_L1_terms = [(cost_to_1[k] if k % 2 == 1 else 0) for k in range(N)]
    val_E0_terms = [(cost_to_0[k] if k % 2 == 0 else 0) for k in range(N)]

    PL_0 = [0] * N
    PE_1 = [0] * N
    PL_1 = [0] * N
    PE_0 = [0] * N

    # N >= 2 is guaranteed by constraints
    PL_0[0] = val_L0_terms[0]
    PE_1[0] = val_E1_terms[0]
    PL_1[0] = val_L1_terms[0]
    PE_0[0] = val_E0_terms[0]
    for k in range(1, N):
        PL_0[k] = PL_0[k-1] + val_L0_terms[k]
        PE_1[k] = PE_1[k-1] + val_E1_terms[k]
        PL_1[k] = PL_1[k-1] + val_L1_terms[k]
        PE_0[k] = PE_0[k-1] + val_E0_terms[k]

    SL_1 = [0] * (N + 1)
    SE_0 = [0] * (N + 1)
    SL_0 = [0] * (N + 1)
    SE_1 = [0] * (N + 1)
    
    for k in range(N - 1, -1, -1):
        SL_1[k] = SL_1[k+1] + val_L1_terms[k]
        SE_0[k] = SE_0[k+1] + val_E0_terms[k]
        SL_0[k] = SL_0[k+1] + val_L0_terms[k]
        SE_1[k] = SE_1[k+1] + val_E1_terms[k]
            
    min_total_cost = float('inf')

    for i in range(N - 1): # i from 0 to N-2
        current_cost_v0 = 0
        current_cost_v1 = 0
        
        if i % 2 == 1: # i is odd
            cost_left_v0 = PL_0[i] + PE_1[i]
            cost_right_v0 = SE_0[i+1] + SL_1[i+1] # Note: order SE_0, SL_1 as per derivation
            current_cost_v0 = cost_left_v0 + cost_right_v0
            
            cost_left_v1 = PL_1[i] + PE_0[i]
            cost_right_v1 = SE_1[i+1] + SL_0[i+1] # Note: order SE_1, SL_0 as per derivation
            current_cost_v1 = cost_left_v1 + cost_right_v1
        else: # i is even
            cost_left_v0 = PE_0[i] + PL_1[i]
            cost_right_v0 = SL_0[i+1] + SE_1[i+1]
            current_cost_v0 = cost_left_v0 + cost_right_v0

            cost_left_v1 = PE_1[i] + PL_0[i]
            cost_right_v1 = SL_1[i+1] + SE_0[i+1]
            current_cost_v1 = cost_left_v1 + cost_right_v1
            
        min_total_cost = min(min_total_cost, current_cost_v0, current_cost_v1)
        
    sys.stdout.write(str(min_total_cost) + "
")

solve()