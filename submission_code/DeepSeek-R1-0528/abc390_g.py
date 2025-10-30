MOD = 998244353
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    if N == 0:
        print(0)
        return
    d_arr = [0] * (N+1)
    for i in range(1, N+1):
        d_arr[i] = len(str(i))
    D_total = sum(d_arr[1:])
    
    fact = [1] * (N+1)
    for i in range(1, N+1):
        fact[i] = fact[i-1] * i % MOD
        
    a_arr = [0] * (N+1)
    for i in range(1, N+1):
        base = pow(10, d_arr[i], MOD)
        a_arr[i] = pow(base, MOD-2, MOD)
    
    max_d = max(d_arr[1:])
    freq = [0] * (max_d+1)
    for i in range(1, N+1):
        d_val = d_arr[i]
        freq[d_val] += 1

    inv_fact = [0] * (N+1)
    inv_fact[N] = pow(fact[N], MOD-2, MOD)
    for i in range(N, 0, -1):
        inv_fact[i-1] = inv_fact[i] * i % MOD

    def nCr(n, r):
        if r < 0 or r > n:
            return 0
        return fact[n] * inv_fact[r] % MOD * inv_fact[n-r] % MOD

    polys = []
    for d_val in range(1, max_d+1):
        if freq[d_val] > 0:
            f = freq[d_val]
            base_a = pow(10, d_val, MOD)
            a_d = pow(base_a, MOD-2, MOD)
            poly = [0] * (f+1)
            for k in range(0, f+1):
                poly[k] = nCr(f, k) * pow(a_d, k, MOD) % MOD
            polys.append(poly)
    
    def conv(a, b):
        n1 = len(a)
        n2 = len(b)
        if n1 == 0 or n2 == 0:
            return []
        size = n1 + n2 - 1
        n = 1
        while n < size:
            n <<= 1
        a_pad = a + [0] * (n - n1)
        b_pad = b + [0] * (n - n2)
        a_ntt = a_pad
        b_ntt = b_pad
        ntt(a_ntt)
        ntt(b_ntt)
        c = [a_ntt[i] * b_ntt[i] % MOD for i in range(n)]
        intt(c)
        return c[:size]

    def ntt(a):
        n = len(a)
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j >= bit:
                j -= bit
                bit >>= 1
            j += bit
            if i < j:
                a[i], a[j] = a[j], a[i]
        step = 2
        while step <= n:
            wn = pow(3, (MOD-1)//step, MOD)
            for i in range(0, n, step):
                w = 1
                for j in range(i, i+step//2):
                    u = a[j]
                    v = a[j+step//2] * w % MOD
                    a[j] = (u + v) % MOD
                    a[j+step//2] = (u - v) % MOD
                    w = w * wn % MOD
            step <<= 1

    def intt(a):
        n = len(a)
        ntt(a)
        a[0] = a[0] * pow(n, MOD-2, MOD) % MOD
        a[1:] = a[:0:-1]
        return a

    Q = [1]
    for poly in polys:
        Q = conv(Q, poly)
    
    U = [0] * N
    for k in range(0, N):
        if k <= N-1:
            U[k] = fact[k] * fact[N-1 - k] % MOD
        else:
            U[k] = 0

    W = conv(Q, U)
    if len(W) > N:
        W = W[:N]
    else:
        W = W + [0] * (N - len(W))
    
    H_arr = [0] * N
    for j in range(0, N):
        H_arr[j] = W[N-1 - j]
    
    from collections import defaultdict
    groups = defaultdict(list)
    for x in range(1, N+1):
        groups[a_arr[x]].append(x)
        
    ans = 0
    base10 = pow(10, D_total, MOD)
    for a_val, xs in groups.items():
        total_x = sum(xs) % MOD
        T_val = 0
        power = 1
        neg_a = (MOD - a_val) % MOD
        for j in range(0, N):
            T_val = (T_val + H_arr[j] * power) % MOD
            power = power * neg_a % MOD
        term = base10 * a_val % MOD * total_x % MOD * T_val % MOD
        ans = (ans + term) % MOD
        
    print(ans % MOD)

if __name__ == '__main__':
    main()