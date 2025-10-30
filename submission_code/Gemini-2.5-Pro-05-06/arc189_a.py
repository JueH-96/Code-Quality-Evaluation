import sys

def solve():
    N = int(sys.stdin.readline())
    A_list = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    if N == 0: # Should not happen based on constraints, but good practice
        print(1)
        return
    if N == 1:
        # X0[0] is 1. A_list[0] must be 1.
        # The problem implies X0[i] = i%2 (1-indexed).
        # So X0_1idx[1] = 1%2 = 1.
        # If A_list[0] (representing A_1) is 1, then 1 way. Else 0 ways.
        # Initial state X0_0idx[i] = (i+1)%2. So X0_0idx[0]=1.
        if A_list[0] == 1:
            print(1)
        else:
            print(0)
        return

    # 0-indexed initial state
    X0 = [(i + 1) % 2 for i in range(N)]

    if A_list[0] != X0[0] or A_list[N-1] != X0[N-1]:
        print(0)
        return

    MAX_N_FIB = N + 5
    fib = [0] * MAX_N_FIB
    if MAX_N_FIB > 1:
        fib[1] = 1
    for i in range(2, MAX_N_FIB):
        fib[i] = (fib[i-1] + fib[i-2]) % MOD
    
    # fib[0]=0, fib[1]=1, fib[2]=1, fib[3]=2, fib[4]=3, ... (standard definition)

    fact = [1] * (N + 1)
    inv_fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i-1] * i) % MOD
    
    inv_fact[N] = pow(fact[N], MOD - 2, MOD)
    for i in range(N - 1, -1, -1): 
        inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD

    def nCr_mod(n_val, r_val):
        if r_val < 0 or r_val > n_val:
            return 0
        num = fact[n_val]
        den = (inv_fact[r_val] * inv_fact[n_val-r_val]) % MOD
        return (num * den) % MOD

    total_ways = 1
    total_ops = 0
    
    i = 0
    while i < N - 1: # iterate through pairs (A_i, A_{i+1})
        if A_list[i] == A_list[i+1]:
            # Start of a run of identical adjacent pairs
            j = i 
            while j < N - 1 and A_list[j] == A_list[j+1]:
                j += 1
            # The run of identical values is A_list[i...j+1]
            # The identical pairs are (A_k, A_{k+1}) for k from i to j-1.
            # Number of such pairs is L_pairs = (j-1) - i + 1 = j - i.
            L_pairs = j - i
            
            ways_seg = fib[L_pairs] # F_L (using L=L_pairs)
            ops_seg = L_pairs // 2

            total_ways = (total_ways * ways_seg) % MOD
            
            current_total_ops = total_ops
            total_ops += ops_seg
            
            total_ways = (total_ways * nCr_mod(total_ops, ops_seg)) % MOD
            
            i = j # The outer loop will increment i, effectively moving to j+1
        else: # A_list[i] != A_list[i+1] (alternating pair)
            if A_list[i] != X0[i] or A_list[i+1] != X0[i+1]:
                # This alternating pair doesn't match X0, impossible
                print(0)
                return
            # This pair is fine, contributes W=1, K=0 to (TotalWays, TotalOps)
            # No change to TotalWays formula structure, just advance i
        i += 1
        
    print(total_ways)

solve()