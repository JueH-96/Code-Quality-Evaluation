import sys
import sys
import sys
def solve():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    MOD = 998244353
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    
    # Precompute f(l, b): number of sequences of length l with initial balance b
    # such that they end with balance 0 and never drop below 0
    # Since we need to compute f(l, b) for l up to N and b up to N
    # f(l, b) = sum_{k} f(l-1, b') where b' = b -1 + V_i and V_i >=1 if b=1 else V_i >=0
    # This is too slow. Alternative Idea:
    # Assume f(l, b) = C(l + b -1, b) - C(l + b -1, b -1)
    # Let's use this formula
    # Precompute factorial and inverse factorial
    MAX = 2*N +2
    factorial = [1]*(MAX)
    for i in range(1, MAX):
        factorial[i] = factorial[i-1]*i % MOD
    inv_fact = [1]*(MAX)
    inv_fact[MAX-1] = pow(factorial[MAX-1], MOD-2, MOD)
    for i in range(MAX-2, -1, -1):
        inv_fact[i] = inv_fact[i+1]*(i+1) % MOD
    def comb(n, k):
        if n <0 or k <0 or k >n:
            return 0
        return factorial[n]*inv_fact[k]%MOD*inv_fact[n-k]%MOD
    f = {}
    # f(l, b) = comb(l + b -1, b) - comb(l + b -1, b-1)
    # precompute all f(l, b)
    # To save memory, we'll compute f(l, b) on the fly when needed
    # Now, implement the DP
    # Initialize DP
    # dp_fixed and dp_less are lists where index b corresponds to balance b
    # To optimize, we'll use two lists and reuse them
    dp_fixed_prev = {}
    dp_less_prev = {}
    # Initial balance is1, and tight=True
    if A[0] >=1:
        dp_fixed_prev[A[0]] =1
    dp_less_prev = {}
    if A[0] >1 and 1 <= A[0]-1 <= N-1:
        dp_less_prev[A[0]-1] =1
    for i in range(1, N):
        dp_fixed_next = {}
        dp_less_next = {}
        a_i = A[i]
        # Transition from fixed
        for b, cnt in dp_fixed_prev.items():
            if b ==1:
                # V_i >=1 and V_i <=a_i
                min_v =1
                max_v = a_i
                if max_v < min_v:
                    continue
                max_v = min(max_v, N -i)
                if max_v < min_v:
                    continue
                # For V_i == a_i
                v = a_i
                b_next = b -1 + v
                if b_next >=1 or (i ==N-1 and b_next ==0):
                    dp_fixed_next[b_next] = (dp_fixed_next.get(b_next,0) + cnt) % MOD
                # For V_i <a_i
                if a_i -1 >=1:
                    v_less = a_i -1
                    b_next_less = b -1 + v_less
                    if b_next_less >=1 or (i ==N-1 and b_next_less ==0):
                        dp_less_next[b_next_less] = (dp_less_next.get(b_next_less,0) + cnt) % MOD
            else:
                # b >1
                # V_i >=0 and V_i <=a_i and b -1 + V_i <=N -i
                min_v =0
                max_v = min(a_i, N -i - (b -1))
                if max_v < min_v:
                    continue
                # For V_i ==a_i
                v = a_i
                b_next = b -1 + v
                if b_next >=1 or (i ==N-1 and b_next ==0):
                    dp_fixed_next[b_next] = (dp_fixed_next.get(b_next,0) + cnt) % MOD
                # For V_i <a_i
                if a_i >0:
                    v_less = a_i -1
                    if v_less >=0:
                        b_next_less = b -1 + v_less
                        if b_next_less >=1 or (i ==N-1 and b_next_less ==0):
                            dp_less_next[b_next_less] = (dp_less_next.get(b_next_less,0) + cnt) % MOD
        # Transition from less
        total_less =0
        for b, cnt in dp_less_prev.items():
            # If b ==1:
            # V_i >=1 and V_i can be up to N -i
            if b ==1:
                min_v =1
                max_v = N -i
                if min_v > max_v:
                    continue
                # All V_i from1 to max_v
                # The number of V_i choices is (max_v - min_v +1)
                # For each V_i, b_next =b -1 +V_i
                # Total counts are sum over V_i, dp_less_prev[b] for each valid b_next
                # This can be computed as:
                # For each V_i, add cnt to b_next
                # But to optimize, iterate V_i and add accordingly
                for v in range(1, max_v +1):
                    b_next = b -1 +v
                    if b_next >=1 or (i ==N-1 and b_next ==0):
                        dp_less_next[b_next] = (dp_less_next.get(b_next,0) + cnt) % MOD
            else:
                # b >1
                # V_i >=0 and b -1 +V_i <=N -i
                min_v =0
                max_v = min(N -i - (b -1), N -i)
                if max_v < min_v:
                    continue
                for v in range(0, max_v +1):
                    b_next = b -1 +v
                    if b_next >=1 or (i ==N-1 and b_next ==0):
                        dp_less_next[b_next] = (dp_less_next.get(b_next,0) + cnt) % MOD
        dp_fixed_prev = dp_fixed_next
        dp_less_prev = dp_less_next
    # After all steps, only sequences with balance 0 are valid
    result =0
    # At the last step, i=N-1, balance must have been1 and V_i=0
    # So we need to check dp_fixed_prev for b_next=0
    # and dp_less_prev should not have balance=0 since only 'fixed' can reach 0
    if 0 in dp_fixed_prev:
        result = (result + dp_fixed_prev[0]) % MOD
    print(result)