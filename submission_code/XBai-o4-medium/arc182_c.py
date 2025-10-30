import sys
import math

MOD = 998244353

def sieve(m):
    if m < 2:
        return []
    is_prime = [True] * (m + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(m**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, m + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    return primes

def multiply_polynomials(a, b):
    res = [0] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            res[i + j] = (res[i + j] + a[i] * b[j]) % MOD
    return res

def compute_initial_S(m_order, t, M):
    initial = []
    for k in range(m_order):
        s = 0
        for m in range(0, k + 1):
            binom = math.comb(2 * t + m - 1, m)
            term = binom * pow(M, m, MOD)
            s = (s + term) % MOD
        initial.append(s)
    return initial

def mat_mult(a, b, mod):
    n = len(a)
    m = len(b[0])
    k_len = len(b)
    res = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for p in range(k_len):
                res[i][j] = (res[i][j] + a[i][p] * b[p][j]) % mod
    return res

def mat_pow(mat, power, mod):
    n = len(mat)
    res = [[0] * n for _ in range(n)]
    for i in range(n):
        res[i][i] = 1
    while power > 0:
        if power % 2 == 1:
            res = mat_mult(res, mat, mod)
        mat = mat_mult(mat, mat, mod)
        power //= 2
    return res

def compute_S(k, rec_coeffs, initial_S, m_order, MOD):
    if k < 0:
        return 0
    if k < m_order:
        return initial_S[k]
    matrix = [[0] * m_order for _ in range(m_order)]
    for i in range(m_order):
        for j in range(m_order):
            if i == 0:
                matrix[0][j] = rec_coeffs[j]
            else:
                if j == i - 1:
                    matrix[i][j] = 1
    steps = k - (m_order - 1)
    mat_pow_result = mat_pow(matrix, steps, MOD)
    initial_vec = initial_S[-m_order:][::-1]
    res = 0
    for i in range(m_order):
        res = (res + mat_pow_result[0][i] * initial_vec[i]) % MOD
    return res

def main():
    N, M = map(int, sys.stdin.readline().split())
    primes = sieve(M)
    if not primes:
        print(0)
        return
    t = len(primes)
    a_p = {p: 0 for p in primes}
    for num in range(1, M + 1):
        temp = num
        for p in primes:
            if p > temp:
                break
            cnt = 0
            while temp % p == 0:
                cnt += 1
                temp = temp // p
            a_p[p] = (a_p[p] + cnt) % MOD
    N_poly = [1]
    for p in primes:
        c = (M - a_p[p]) % MOD
        new_poly = [0] * (len(N_poly) + 1)
        for i in range(len(N_poly)):
            new_poly[i] = (new_poly[i] + N_poly[i]) % MOD
            new_poly[i + 1] = (new_poly[i + 1] - N_poly[i] * c) % MOD
        N_poly = new_poly
    exponent = 2 * t
    D_poly = [0] * (exponent + 1)
    for k in range(exponent + 1):
        binom = math.comb(exponent, k)
        sign = (-1) ** k
        term = binom * sign * (M ** k)
        D_poly[k] = term % MOD
    max_len = max(len(N_poly), len(D_poly))
    Q_poly = [0] * max_len
    for i in range(max_len):
        val_N = N_poly[i] if i < len(N_poly) else 0
        val_D = D_poly[i] if i < len(D_poly) else 0
        Q_poly[i] = (val_N - val_D) % MOD
    poly_P = multiply_polynomials(D_poly, [1, -1])
    m_order = len(poly_P) - 1
    rec_coeffs = [(-poly_P[i]) % MOD for i in range(1, m_order + 1)]
    initial_S = compute_initial_S(m_order, t, M)
    total = 0
    for i in range(len(Q_poly)):
        q_i = Q_poly[i]
        if q_i == 0:
            continue
        k = N - i
        if k < 0:
            continue
        s_k = compute_S(k, rec_coeffs, initial_S, m_order, MOD)
        total = (total + q_i * s_k) % MOD
    print(total % MOD)

if __name__ == '__main__':
    main()