import sys

MOD = 998244353

# ---------- helpers ----------
def mod_inv(x: int) -> int:
    return pow(x, MOD - 2, MOD)


# ---------- obtain primes up to 16 ----------
ALL_PRIMES = [2, 3, 5, 7, 11, 13]


# ---------- μ_U for every subset ----------
def calc_mu(M: int, primes: list) -> list:
    r = len(primes)
    size = 1 << r
    mu = [0] * size
    invM = mod_inv(M)

    # exponent table
    exps = [[0] * r for _ in range(M + 1)]
    for v in range(1, M + 1):
        tmp = v
        for idx, p in enumerate(primes):
            cnt = 0
            while tmp % p == 0:
                tmp //= p
                cnt += 1
            exps[v][idx] = cnt
        # tmp is 1 afterwards (M<=16)

    for mask in range(1, size):
        s = 0
        for v in range(1, M + 1):
            prod = 1
            for idx in range(r):
                if mask & (1 << idx):
                    e = exps[v][idx]
                    if e == 0:
                        prod = 0
                        break
                    prod *= e
            s += prod
        mu[mask] = (s % MOD) * invM % MOD
    mu[0] = 1
    return mu


# ---------- c_m via DP on set partitions ----------
def calc_c(mu: list, r: int) -> list:
    size = 1 << r
    dp = [[0] * (r + 1) for _ in range(size)]
    dp[0][0] = 1

    for mask in range(1, size):
        lowbit = mask & -mask
        sub = mask
        while sub:
            if sub & lowbit:  # ensure every partition counted once
                rest = mask ^ sub
                for m in range(1, r + 1):
                    if dp[rest][m - 1]:
                        dp[mask][m] = (dp[mask][m] + dp[rest][m - 1] * mu[sub]) % MOD
            sub = (sub - 1) & mask

    c = [0] * (r + 1)
    for mask in range(size):
        for m in range(r + 1):
            c[m] = (c[m] + dp[mask][m]) % MOD
    return c  # c[0] is 1


# ---------- matrix utilities ----------
def mat_mul(A, B):
    n = len(A)
    res = [[0] * n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ri = res[i]
        for k, v in enumerate(Ai):
            if v:
                Bk = B[k]
                for j in range(n):
                    Ri[j] = (Ri[j] + v * Bk[j]) % MOD
    return res


def mat_pow(mat, power: int):
    n = len(mat)
    res = [[0] * n for _ in range(n)]
    for i in range(n):
        res[i][i] = 1
    while power:
        if power & 1:
            res = mat_mul(res, mat)
        mat = mat_mul(mat, mat)
        power >>= 1
    return res


def mat_vec_mul(mat, vec):
    n = len(mat)
    res = [0] * n
    for i in range(n):
        s = 0
        row = mat[i]
        for j in range(n):
            s += row[j] * vec[j]
        res[i] = s % MOD
    return res


# ---------- build transition matrix ----------
def build_matrix(q: int, r: int):
    d = 2 * (r + 1)
    T = [[0] * d for _ in range(d)]

    # a-part
    # index mapping
    # a_j : 0 .. r
    # s_j : r+1 .. 2r+1
    for j in range(r + 1):
        if j == 0:
            T[0][0] = q % MOD
        else:
            T[j][j] = q % MOD
            T[j][j - 1] = (q * j) % MOD

    # s-part
    for j in range(r + 1):
        row = r + 1 + j
        # keep previous sum
        T[row][row] = 1
        # add new a' contributions (same as entries in corresponding a-row)
        if j == 0:
            T[row][0] = (T[0][0])  # q
        else:
            T[row][j] = T[j][j]           # q
            T[row][j - 1] = (T[row][j - 1] + T[j][j - 1]) % MOD  # q * j
    return T


# ---------- main ----------
def main():
    N_str, M_str = sys.stdin.readline().split()
    N = int(N_str)
    M = int(M_str)

    if M == 1:
        # every product is 1, divisor count is 1
        print(N % MOD)
        return

    # relevant primes
    primes = [p for p in ALL_PRIMES if p <= M]
    r = len(primes)

    mu = calc_mu(M, primes)
    c = calc_c(mu, r)

    q = M % MOD
    d = 2 * (r + 1)

    T = build_matrix(q, r)
    T_pow = mat_pow(T, N)

    # initial vector
    vec0 = [0] * d
    vec0[0] = 1  # a_0^0 = 1

    vecN = mat_vec_mul(T_pow, vec0)

    # s_N^m are at positions r+1 + m
    ans = 0
    for m in range(r + 1):
        sNm = vecN[r + 1 + m]
        ans = (ans + c[m] * sNm) % MOD

    print(ans % MOD)


if __name__ == "__main__":
    main()