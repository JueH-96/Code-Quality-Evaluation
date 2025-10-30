MOD = 998244353

def precompute_factorials(max_n, mod=MOD):
    fact = [1] * (max_n + 1)
    inv_fact = [1] * (max_n + 1)
    for i in range(2, max_n + 1):
        fact[i] = fact[i - 1] * i % mod
    inv_fact[max_n] = pow(fact[max_n], mod - 2, mod)
    for i in range(max_n, 0, -1):
        inv_fact[i - 1] = inv_fact[i] * i % mod
    return fact, inv_fact

max_N = 3000000
fact_global, inv_fact_global = precompute_factorials(max_N, MOD)

def nCr(n, r):
    if r < 0 or r > n:
        return 0
    res = fact_global[n] * inv_fact_global[r] % MOD
    res = res * inv_fact_global[n - r] % MOD
    return res

def main():
    import sys
    data = sys.stdin.read().split()
    W = int(data[0]); H = int(data[1]); L = int(data[2]); R = int(data[3]); D = int(data[4]); U = int(data[5])
    
    total = 0
    
    sc1 = nCr(W - R + H + 3, H + 2)
    part_c1 = (sc1 - (H + 3)) % MOD
    part_c1 = (part_c1 - (W - R) * (H + 2)) % MOD
    SC = part_c1

    term_d1 = nCr(R - L + H - U + 2, H - U)
    part_d1 = (term_d1 - (H - U + 1)) % MOD
    part_d1 = (part_d1 - (R - L + 1) * (H - U + 1)) % MOD

    term_d2 = nCr(W - R + H - U + 1, H - U)
    part_d2 = (term_d2 - 1 - (H - U)) % MOD
    part_d2 = (R - L + 1) * part_d2 % MOD
    SD = (part_d1 + part_d2) % MOD

    part_b1 = nCr(R - L + D + 3, D + 1) - (D + 2)
    part_b1 = (part_b1 - (R - L + 1) * (D + 1)) % MOD

    part_b2 = 0
    for y in range(0, D):
        f_c_val = nCr((W - R - 1) + (H - y) + 2, (H - y) + 1) - 1
        c_val = nCr(R - L + y + 1, y + 1)
        part_b2 = (part_b2 + f_c_val * c_val) % MOD
    SB = (part_b1 + part_b2) % MOD

    part_a1 = nCr(L + H + 3, H + 2) - L * (H + 2) - (H + 3)
    part_a1 %= MOD

    B_list = [0] * D
    for y in range(0, D):
        B_list[y] = nCr((W - R - 1) + (H - y) + 2, (H - y) + 1) - 1
        B_list[y] %= MOD

    suffix_B = [0] * (D + 1)
    for i in range(D - 1, -1, -1):
        suffix_B[i] = (suffix_B[i + 1] + B_list[i]) % MOD

    Q_list = [0] * D
    for j in range(0, D):
        Q_list[j] = nCr(R - L + j, R - L)

    sum_f = 0
    for j in range(0, D):
        sum_f = (sum_f + Q_list[j] * suffix_B[j]) % MOD

    if D - 1 < 0:
        sum_t = 0
    else:
        sum_t = nCr(R - L + D + 1, D - 1) - 1 - D
    sum_g = (sum_t + sum_f) % MOD
    part_a2 = L * sum_g % MOD
    SA = (part_a1 + part_a2) % MOD

    total = (SA + SB + SC + SD) % MOD
    print(total)

if __name__ == '__main__':
    main()