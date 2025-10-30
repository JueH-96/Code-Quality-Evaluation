# YOUR CODE HERE
import sys
import sys
def solve():
    import sys
    import sys
    MOD = 998244353
    K_and_rest = sys.stdin.read().split()
    K = int(K_and_rest[0])
    C = list(map(int, K_and_rest[1:27]))
    
    # Precompute factorial and inverse factorial
    fact = [1] * (K + 1)
    for i in range(1, K +1):
        fact[i] = fact[i-1] * i % MOD
    inv_fact = [1] * (K +1)
    inv_fact[K] = pow(fact[K], MOD-2, MOD)
    for i in range(K-1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
    
    # Initialize G
    G = [0] * (K +1)
    G[0] = 1

    for Ci in C:
        if Ci > K:
            Ci = K
        new_G = [0] * (K +1)
        for l in range(0, K +1):
            if G[l] ==0:
                continue
            max_k = min(Ci, K - l)
            # Instead of looping k, we can precompute and add
            # For better performance, use range
            # but in Python, it's still somewhat slow
            # Thus, use list slicing
            # However, we need to loop k
            for k in range(0, max_k +1):
                new_G[l +k] = (new_G[l +k] + G[l] * inv_fact[k]) % MOD
        G = new_G

    # Precompute powers of 1 (since 26^0=1 is already handled in the convolution)
    # Now compute the sum of a_l from 1 to K
    total =0
    for l in range(1, K +1):
        total = (total + G[l] * fact[l]) % MOD
    print(total)