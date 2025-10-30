import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    # A_i is 1-indexed, A[0] is a dummy value. A_i for i from 2 to N.
    A = [0] + list(map(int, sys.stdin.readline().split())) 

    MOD = 998244353

    # Precompute factorials: fact[k] = k!
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i-1] * i) % MOD

    # Precompute modular inverses: inv[k] = k^(-1) mod MOD
    inv = [0] * (N + 1)
    inv[1] = 1 
    for i in range(2, N + 1):
        # Using Fermat's Little Theorem or extended Euclidean algorithm equivalent:
        # inv[i] = pow(i, MOD - 2, MOD)
        # Or iterative calculation:
        inv[i] = (MOD - (MOD // i) * inv[MOD % i] % MOD) % MOD
    
    # Precompute prefix sums for A_j * inv[j]
    # prefix_sum_A_over_i[k] stores sum_{j=2 to k} (A_j * inv[j])
    prefix_sum_A_over_i = [0] * (N + 1)
    for i in range(2, N + 1):
        prefix_sum_A_over_i[i] = (prefix_sum_A_over_i[i-1] + A[i] * inv[i]) % MOD

    results = []
    for _ in range(Q):
        u, v = map(int, sys.stdin.readline().split())
        
        # Ensure u < v for consistency with formula derivation
        if u > v:
            u, v = v, u

        ans = 0
        if u == 1:
            # Special case for u = 1.
            # SumDist(1,v) = (N-1)! * [ sum_{i=2}^{v-1} A_i * (1/i) + A_v ]
            # The sum term is prefix_sum_A_over_i[v-1]
            term_sum_mid = prefix_sum_A_over_i[v-1]
            term_v = A[v]
            ans = (fact[N-1] * (term_sum_mid + term_v)) % MOD
        else:
            # General case for u > 1.
            # SumDist(u,v) = (N-1)! * [ A_u * (1 - 1/u) + sum_{i=u+1}^{v-1} A_i * (1/i) + A_v ]
            
            # Term for A_u * (1 - 1/u)
            term_u = (A[u] * (1 - inv[u] + MOD)) % MOD
            
            # Term for sum_{i=u+1}^{v-1} A_i * (1/i)
            # This is prefix_sum_A_over_i[v-1] - prefix_sum_A_over_i[u]
            # Need to handle potential negative result from subtraction before modulo
            term_sum_mid = (prefix_sum_A_over_i[v-1] - prefix_sum_A_over_i[u] + MOD) % MOD
            
            # Term for A_v
            term_v = A[v]
            
            total_prob_sum = (term_u + term_sum_mid + term_v) % MOD
            
            ans = (fact[N-1] * total_prob_sum) % MOD
        
        results.append(str(ans))
    
    sys.stdout.write('
'.join(results) + '
')

solve()