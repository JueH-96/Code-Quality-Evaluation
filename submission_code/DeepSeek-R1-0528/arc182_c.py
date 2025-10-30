MOD = 998244353

def mat_mult(A, B):
    n = len(A)
    p = len(B)
    m = len(B[0])
    C = [[0] * m for _ in range(n)]
    for i in range(n):
        for k in range(p):
            if A[i][k]:
                for j in range(m):
                    if B[k][j]:
                        C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % MOD
    return C

def mat_pow(matrix, power):
    n = len(matrix)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = matrix
    while power:
        if power & 1:
            result = mat_mult(result, base)
        base = mat_mult(base, base)
        power //= 2
    return result

def sum_geometric(a, n):
    if a == 0:
        return 0
    if a == 1:
        return n % MOD
    an = pow(a, n + 1, MOD)
    inv_a_minus_1 = pow(a - 1, MOD - 2, MOD)
    return (an - 1) * inv_a_minus_1 % MOD

def sum_geometric0(a, n):
    if a == 0:
        return 0
    if a == 1:
        return (n + 1) % MOD
    an = pow(a, n + 1, MOD)
    inv_a_minus_1 = pow(a - 1, MOD - 2, MOD)
    return (an - 1) * inv_a_minus_1 % MOD

def falling_factorial_sum(t, N, M_val):
    if t == 0:
        return sum_geometric(M_val, N)
    if M_val == 0:
        return 0
    if M_val == 1:
        return (N - t + 1) % MOD
    inv_M = pow(M_val, MOD - 2, MOD)
    coeffs = [0] * (t + 1)
    coeffs[0] = 1
    for i in range(1, t + 1):
        new_coeffs = [0] * (t + 1)
        for j in range(i + 1):
            if j < i:
                new_coeffs[j] = (new_coeffs[j] + coeffs[j] * (1 - 2 * i)) % MOD
            if j > 0:
                new_coeffs[j] = (new_coeffs[j] + coeffs[j - 1] * i) % MOD
        coeffs = new_coeffs
    T = t
    M_inv_power = pow(inv_M, T, MOD)
    poly = 0
    for deg, co in enumerate(coeffs):
        poly = (poly + co * pow(N, deg, MOD)) % MOD
    return poly * M_inv_power % MOD

def solve(N, M_val):
    primes = [2, 3, 5, 7, 11, 13]
    num_primes = len(primes)
    v = [[0] * num_primes for _ in range(M_val + 1)]
    for a in range(1, M_val + 1):
        temp = a
        for j, p in enumerate(primes):
            cnt = 0
            while temp % p == 0:
                cnt += 1
                temp //= p
            v[a][j] = cnt
    c_total = [0] * (1 << num_primes)
    for mask in range(1 << num_primes):
        total = 0
        for a in range(1, M_val + 1):
            prod = 1
            for j in range(num_primes):
                if mask & (1 << j):
                    prod = prod * v[a][j]
            total = (total + prod) % MOD
        c_total[mask] = total
    total_ans = 0
    for S_mask in range(1 << num_primes):
        s = bin(S_mask).count('1')
        bits = []
        for j in range(num_primes):
            if S_mask & (1 << j):
                bits.append(j)
        abs_mask = [0] * (1 << s)
        for rel_mask in range(1 << s):
            am = 0
            for i in range(s):
                if rel_mask & (1 << i):
                    am |= (1 << bits[i])
            abs_mask[rel_mask] = am
        full_rel = (1 << s) - 1
        dp = [[0] * (1 << s) for _ in range(s + 1)]
        dp[0][0] = 1
        for t in range(1, s + 1):
            for rel in range(1, 1 << s):
                u = rel
                while u:
                    u_rel = u
                    u_abs = abs_mask[u_rel]
                    rem = rel ^ u_rel
                    dp[t][rel] = (dp[t][rel] + c_total[u_abs] * dp[t - 1][rem]) % MOD
                    u = (u - 1) & rel
        t_max = min(s, N)
        term_S = 0
        for t in range(0, t_max + 1):
            if t == 0:
                inner_sum = falling_factorial_sum(0, N, M_val)
            else:
                inner_sum = falling_factorial_sum(t, N, M_val)
            term_S = (term_S + dp[t][full_rel] * inner_sum) % MOD
        total_ans = (total_ans + term_S) % MOD
    return total_ans

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    M_val = int(data[1])
    if M_val == 0:
        print(0)
        return
    print(solve(N, M_val) % MOD)

if __name__ == "__main__":
    main()