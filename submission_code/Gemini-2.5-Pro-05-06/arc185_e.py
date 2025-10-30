import sys

def solve():
    N = int(sys.stdin.readline())
    A_input = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    MAX_A_VAL = 0
    if N > 0:
        # Constraints: A_i >= 1, so max(A_input) will be >= 1
        MAX_A_VAL = A_input[0]
        for i in range(1, N):
            if A_input[i] > MAX_A_VAL:
                MAX_A_VAL = A_input[i]
    
    phi = list(range(MAX_A_VAL + 1))
    # phi[0] is not used. phi[1] is 1 by default from range.
    for i in range(2, MAX_A_VAL + 1):
        if phi[i] == i: # i is prime
            for j in range(i, MAX_A_VAL + 1, i):
                phi[j] -= phi[j] // i
    
    pow2 = [0] * N 
    if N > 0:
        pow2[0] = 1
    for i in range(1, N):
        pow2[i] = (pow2[i-1] * 2) % MOD

    divs = [[] for _ in range(MAX_A_VAL + 1)]
    for i in range(1, MAX_A_VAL + 1):
        for j in range(i, MAX_A_VAL + 1, i): 
            divs[j].append(i)

    val_for_d = [0] * (MAX_A_VAL + 1)
    
    ans_prev_m = 0 
    
    results_output = [] # To store results before printing

    for m_idx in range(N): 
        current_A_val = A_input[m_idx]
        
        term_T_m = 0
        if m_idx > 0: # For m_idx=0 (m=1), T_1 = 0. Sum is empty.
            for d in divs[current_A_val]:
                term_T_m = (term_T_m + phi[d] * val_for_d[d]) % MOD
        
        ans_current_m = (2 * ans_prev_m + term_T_m) % MOD
        results_output.append(str(ans_current_m))
        
        val_to_add_for_A_m_idx = pow2[m_idx]
        for d in divs[current_A_val]:
            val_for_d[d] = (val_for_d[d] + val_to_add_for_A_m_idx) % MOD
        
        ans_prev_m = ans_current_m
            
    sys.stdout.write("
".join(results_output) + "
")

solve()