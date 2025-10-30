MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    W = int(input[0])
    H = int(input[1])
    L = int(input[2])
    R = int(input[3])
    D = int(input[4])
    U = int(input[5])

    max_n = 2 * 10**6 + 10
    fact = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = fact[i-1] * i % MOD

    inv_fact = [1] * (max_n + 1)
    inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
    for i in range(max_n - 1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

    def comb(n, k):
        if n < 0 or k < 0 or n < k:
            return 0
        return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD

    def compute_S_A(L, H):
        if L == 0:
            return 0
        n = H + L + 3
        k = L + 1
        c = comb(n, k)
        sum_part = (c - (H + 3)) % MOD
        res = (sum_part - L * (H + 2)) % MOD
        return res

    def compute_S_B(W, R, H):
        M = W - R
        if M == 0:
            return 0
        a = R + H + 4
        b = H + 1
        c1 = comb(a + M, b + 1)
        c2 = comb(a, b + 1)
        sum_part = (c1 - c2) % MOD
        res = (sum_part - M * (H + 2)) % MOD
        return res

    def compute_S_C(W, D):
        if D == 0:
            return 0
        n = W + D + 2
        k = D + 1
        c = comb(n, k)
        sum_part = (c - (W + 3)) % MOD
        res = (sum_part - D * (W + 2)) % MOD
        return res

    def compute_S_D(W, H, U):
        if U >= H:
            return 0
        # Similar to S_C but for y > U
        # y ranges from U+1 to H
        # which is equivalent to y' = y - (U+1), new H' = H - (U+1)
        # but need to derive formula
        # For S_D, it's sum_{y=U+1}^H ... which is sum_{y=0}^{H - (U+1)} ... with W and H - (U+1)
        new_H = H - (U + 1)
        if new_H < 0:
            return 0
        n = W + new_H + 3
        k = new_H + 1
        c = comb(n, k)
        sum_part = (c - (W + 3)) % MOD
        res = (sum_part - (new_H + 1) * (W + 2)) % MOD
        return res

    def compute_S_AC(L, H, D):
        if L == 0 or D == 0:
            return 0
        n = L + D + 2
        k = L + 1
        c = comb(n, k)
        sum_part = (c - (D + 2)) % MOD
        res = (sum_part - L * (D + 1)) % MOD
        return res

    def compute_S_AD(L, H, U):
        if L == 0 or U >= H:
            return 0
        new_H = H - (U + 1)
        if new_H < 0:
            return 0
        n = L + new_H + 3
        k = L + 1
        c = comb(n, k)
        sum_part = (c - (new_H + 3)) % MOD
        res = (sum_part - L * (new_H + 2)) % MOD
        return res

    def compute_S_BC(R, W, D):
        if R >= W or D == 0:
            return 0
        M = W - R
        a = R + D + 4
        b = D + 1
        c1 = comb(a + M, b + 1)
        c2 = comb(a, b + 1)
        sum_part = (c1 - c2) % MOD
        res = (sum_part - M * (D + 2)) % MOD
        return res

    def compute_S_BD(R, W, H, U):
        if R >= W or U >= H:
            return 0
        M = W - R
        new_H = H - (U + 1)
        if new_H < 0:
            return 0
        a = R + new_H + 4
        b = new_H + 1
        c1 = comb(a + M, b + 1)
        c2 = comb(a, b + 1)
        sum_part = (c1 - c2) % MOD
        res = (sum_part - M * (new_H + 2)) % MOD
        return res

    S_A = compute_S_A(L, H)
    S_B = compute_S_B(W, R, H)
    S_C = compute_S_C(W, D)
    S_D = compute_S_D(W, H, U)

    S_AC = compute_S_AC(L, H, D)
    S_AD = compute_S_AD(L, H, U)
    S_BC = compute_S_BC(R, W, D)
    S_BD = compute_S_BD(R, W, H, U)

    total = (S_A + S_B + S_C + S_D - S_AC - S_AD - S_BC - S_BD) % MOD
    print(total)

if __name__ == "__main__":
    main()