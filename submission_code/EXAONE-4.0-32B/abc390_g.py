import sys
mod = 998244353

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    if N == 0:
        print(0)
        return
        
    max_n = 200000
    fact = [1] * (max_n + 1)
    inv_fact = [1] * (max_n + 1)
    
    for i in range(1, max_n + 1):
        fact[i] = fact[i - 1] * i % mod
        
    inv_fact[max_n] = pow(fact[max_n], mod - 2, mod)
    for i in range(max_n, 0, -1):
        inv_fact[i - 1] = inv_fact[i] * i % mod

    def ntt(a, inverse=False):
        n = len(a)
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j |= bit
            if i < j:
                a[i], a[j] = a[j], a[i]
        step = 2
        while step <= n:
            wn = pow(3, (mod - 1) // step, mod)
            if inverse:
                wn = pow(wn, mod - 2, mod)
            for i in range(0, n, step):
                w = 1
                for j1 in range(i, i + step // 2):
                    j2 = j1 + step // 2
                    u = a[j1]
                    v = w * a[j2] % mod
                    a[j1] = (u + v) % mod
                    a[j2] = (u - v) % mod
                    w = w * wn % mod
            step <<= 1
        if inverse:
            inv_n = pow(n, mod - 2, mod)
            for i in range(n):
                a[i] = a[i] * inv_n % mod

    def convolve(a, b):
        n1 = len(a)
        n2 = len(b)
        total_length = n1 + n2 - 1
        n = 1
        while n < total_length:
            n <<= 1
        a_ntt = a + [0] * (n - n1)
        b_ntt = b + [0] * (n - n2)
        ntt(a_ntt)
        ntt(b_ntt)
        c_ntt = [a_ntt[i] * b_ntt[i] % mod for i in range(n)]
        ntt(c_ntt, inverse=True)
        return c_ntt[:total_length]

    max_digits = len(str(N))
    cnt = [0] * (max_digits + 1)
    s_list = [0] * (max_digits + 1)
    for d in range(1, max_digits + 1):
        low = 10**(d - 1)
        high = min(10**d - 1, N)
        if low > N:
            continue
        cnt[d] = high - low + 1
        s_list[d] = (low + high) * cnt[d] // 2

    P = [1]
    for d in range(1, max_digits + 1):
        c = cnt[d]
        if c == 0:
            continue
        w_val = pow(10, d, mod)
        poly_d = [0] * (c + 1)
        for i in range(c + 1):
            poly_d[i] = fact[c] * inv_fact[i] % mod * inv_fact[c - i] % mod * pow(w_val, i, mod) % mod
        P_new = convolve(P, poly_d)
        if len(P_new) > N + 1:
            P_new = P_new[:N + 1]
        P = P_new

    q = [0] * (N + 1)
    for d in range(1, max_digits + 1):
        w_val = pow(10, d, mod)
        s_val = s_list[d] % mod
        for t in range(N + 1):
            term = s_val * w_val % mod * pow(-w_val, t, mod) % mod
            q[t] = (q[t] + term) % mod

    Q_prime = [0] + q

    R = convolve(P, Q_prime)
    G = R[:N + 1]

    total_sum = (N * (N + 1) // 2) % mod

    ans = 0
    for j in range(1, N + 1):
        k = N - j
        Tj = (total_sum * P[k] - G[k]) % mod
        term = fact[j - 1] * fact[N - j] % mod * Tj % mod
        ans = (ans + term) % mod

    ans %= mod
    if ans < 0:
        ans += mod
    print(ans)

if __name__ == '__main__':
    main()