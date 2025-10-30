import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    MOD = 998244353
    N = int(sys.stdin.readline())

    # Precompute lengths and powers of 10
    len_ = [0] * (N + 1)
    for x in range(1, N + 1):
        len_[x] = len(str(x))
    
    inv10 = pow(10, MOD-2, MOD)
    v = [0] * (N + 1)  # v[x] = 10^{-len[x]} mod MOD
    pow10 = [1] * (N + 1)  # 10^{total_rest[x]} mod MOD
    total_len = 0
    for x in range(1, N + 1):
        l = len_[x]
        total_len += l
        v[x] = pow(inv10, l, MOD)
        pow10[x] = pow(10, total_len - l, MOD)  # total_rest[x] = sum of len[y] for y !=x

    # Compute factorial and inverse factorial arrays
    max_n = N + 2
    fact = [1] * (max_n)
    for i in range(1, max_n):
        fact[i] = fact[i-1] * i % MOD
    inv_fact = [1] * (max_n)
    inv_fact[max_n-1] = pow(fact[max_n-1], MOD-2, MOD)
    for i in range(max_n-2, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

    # Compute polynomial P = product_{x=1..N} (1 + v[x] * X)
    # Using dynamic programming to build ESS (elementary symmetric sums)
    # ESS[k] is the sum of all products of v[x] over subsets of size k
    ESS = [0] * (N + 1)
    ESS[0] = 1
    for x in range(1, N + 1):
        for k in range(x, 0, -1):
            ESS[k] = (ESS[k] + ESS[k-1] * v[x]) % MOD

    # Now, for each x, compute S[x] = sum_{k=0}^{N-1} k! * (N -k -1)! * ESS_k_notx
    # where ESS_k_notx = ESS_k - v[x] * ESS_{k-1}
    total = 0
    sum_w = 0
    for x in range(1, N + 1):
        w_x = x * pow(pow10[x], 1, MOD) % MOD
        res = 0
        for k in range(0, N):
            if k <= N:
                if k == 0:
                    ess_k = 1 if k == 0 else 0
                else:
                    ess_k = ESS[k]
                if k == 0:
                    ess_km1 = 0
                else:
                    ess_km1 = ESS[k-1] if k-1 >=0 else 0
                # ESS_k_notx = ESS_k - v[x] * ESS_{k-1}
                term = (ess_k - v[x] * ess_km1) % MOD
                term = term * fact[k] % MOD
                term = term * fact[N - k - 1] % MOD
                res = (res + term) % MOD
        contrib = w_x * res % MOD
        total = (total + contrib) % MOD

    # Multiply by the total_product
    total_product = 1
    for x in range(1, N + 1):
        total_product = total_product * pow(10, len_[x], MOD) % MOD

    ans = total * total_product % MOD
    print(ans)

threading.Thread(target=main).start()