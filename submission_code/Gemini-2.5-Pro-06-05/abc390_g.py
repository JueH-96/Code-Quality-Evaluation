import sys

def main():
    # Set recursion limit for deep product tree, although iterative multiplication will be used
    sys.setrecursionlimit(2 * 10**5 + 5)

    N = int(sys.stdin.readline())
    MOD = 998244353

    # Fast power
    def power(a, b):
        res = 1
        a %= MOD
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % MOD
            a = (a * a) % MOD
            b //= 2
        return res

    # Modular inverse
    def inv(n):
        return power(n, MOD - 2)

    # Precompute factorials and their inverses
    fact = [1] * (N + 1)
    inv_fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    inv_fact[N] = inv(fact[N])
    for i in range(N - 1, -1, -1):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

    # Combinations
    def nCr_mod(n, r):
        if r < 0 or r > n:
            return 0
        num = fact[n]
        den = (inv_fact[r] * inv_fact[n - r]) % MOD
        return (num * den) % MOD

    # NTT implementation
    g = 3 # Primitive root for MOD
    def ntt(a, invert):
        n = len(a)
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                a[i], a[j] = a[j], a[i]

        length = 2
        while length <= n:
            wlen = power(g, (MOD - 1) // length)
            if invert:
                wlen = inv(wlen)
            i = 0
            while i < n:
                w = 1
                for j in range(length // 2):
                    u = a[i + j]
                    v = (a[i + j + length // 2] * w) % MOD
                    a[i + j] = (u + v) % MOD
                    a[i + j + length // 2] = (u - v + MOD) % MOD
                    w = (w * wlen) % MOD
                i += length
            length <<= 1
        
        if invert:
            n_inv = inv(n)
            for i in range(n):
                a[i] = (a[i] * n_inv) % MOD

    def multiply(p1, p2):
        if not p1 or not p2:
            return []
        deg1 = len(p1) - 1
        deg2 = len(p2) - 1
        res_deg = deg1 + deg2
        
        n = 1
        while n <= res_deg:
            n <<= 1
            
        p1_pad = p1 + [0] * (n - len(p1))
        p2_pad = p2 + [0] * (n - len(p2))
        
        ntt(p1_pad, False)
        ntt(p2_pad, False)
        
        res = [(p1_pad[i] * p2_pad[i]) % MOD for i in range(n)]
        
        ntt(res, True)
        
        return res[:res_deg + 1]

    # Main logic
    
    # 1. Group numbers by number of digits
    d_counts = []
    d_sums = []
    d_W = []
    
    low = 1
    p10 = 10
    inv2 = inv(2)
    
    while low <= N:
        high = min(N, p10 - 1)
        
        n_d = high - low + 1
        sum_d = (((high + low) % MOD) * (n_d % MOD)) % MOD
        sum_d = (sum_d * inv2) % MOD
        
        d_counts.append(n_d)
        d_sums.append(sum_d)
        d_W.append(p10)
        
        low = p10
        p10 *= 10
        
    # 2. Compute P_d(t) = (1 + W_d * t)^n_d
    polys = []
    for i in range(len(d_counts)):
        n_d = d_counts[i]
        W_d = d_W[i]
        
        p_d = [0] * (n_d + 1)
        W_pow = 1
        for j in range(n_d + 1):
            p_d[j] = (nCr_mod(n_d, j) * W_pow) % MOD
            W_pow = (W_pow * W_d) % MOD
        polys.append(p_d)

    # 3. Compute P(t) = product of all P_d(t)
    if not polys:
      print(0)
      return

    # Iterative multiplication is robust
    P_all = [1]
    for p in polys:
        P_all = multiply(P_all, p)

    # 4. Compute F(t) = sum_d Sum_d * (P(t) / (1 + W_d * t))
    F_coeffs = [0] * N
    
    for i in range(len(d_counts)):
        Sum_d = d_sums[i]
        W_d = d_W[i]
        
        inv_Wd = inv(W_d)
        neg_inv_Wd = (MOD - inv_Wd) % MOD
        
        q = [0] * (len(P_all) - 1)
        if not q:
            continue
            
        q[-1] = P_all[-1]
        for j in range(len(P_all) - 2, 0, -1):
            q[j-1] = (P_all[j] + q[j] * neg_inv_Wd) % MOD

        # Scale by inv_Wd
        for j in range(len(q)):
            q[j] = (q[j] * inv_Wd) % MOD
            
        for j in range(len(q)):
            F_coeffs[j] = (F_coeffs[j] + Sum_d * q[j]) % MOD

    # 5. Final summation
    total_sum = 0
    for j in range(N):
        term = (fact[N - 1 - j] * fact[j]) % MOD
        term = (term * F_coeffs[j]) % MOD
        total_sum = (total_sum + term) % MOD
        
    print(total_sum)

if __name__ == "__main__":
    main()