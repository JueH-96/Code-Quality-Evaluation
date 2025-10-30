import sys

MOD = 998244353
PRIMITIVE_ROOT = 3               # primitive root of MOD
BIT = 20                         # numbers are < 2**20
SIZE = 1 << BIT                  # 2**20

# ----------  Walsh–Hadamard transform (XOR)  ----------
def fwht_int(a):                 # transform over plain integers
    n = len(a)
    step = 1
    while step < n:
        jump = step << 1
        for i in range(0, n, jump):
            for j in range(i, i + step):
                x = a[j]
                y = a[j + step]
                a[j] = x + y
                a[j + step] = x - y
        step = jump


def fwht_mod(a, mod):            # transform modulo `mod`
    n = len(a)
    step = 1
    while step < n:
        jump = step << 1
        for i in range(0, n, jump):
            for j in range(i, i + step):
                x = a[j]
                y = a[j + step]
                a[j] = (x + y) % mod
                a[j + step] = (x - y) % mod
        step = jump


# -------------------------------------------------------
def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(N)]

    # ---------- frequency array of A -------------------
    freq = [0] * SIZE
    for v in A:
        freq[v] += 1

    # ---------- transform of frequency -----------------
    freq_hat = freq[:]           # copy, we will transform in-place
    fwht_int(freq_hat)           # values now in range [-N .. N]

    # ---------- compute n1(u)  -------------------------
    #  n1(u) = (N - T(u)) // 2,  where  T(u) = freq_hat[u]
    n1 = [(N - t) >> 1 for t in freq_hat]   # length SIZE

    # ---------- prepare f(x) = x^K mod MOD ------------
    pow_table = [pow(x, K, MOD) for x in range(SIZE)]

    # ---------- transform of f -------------------------
    fwht_mod(pow_table, MOD)      # pow_table now holds F̂(u)

    # ---------- accumulate by k = n1(u) ----------------
    sum_by_k = [0] * (N + 1)      # coefficient array
    for k_val, fh in zip(n1, pow_table):
        sum_by_k[k_val] = (sum_by_k[k_val] + fh) % MOD

    # ---------- pre-compute inverses -------------------
    inv_size = pow(SIZE, MOD - 2, MOD)      # 1/2^BIT  (for Parseval)
    inv_M     = pow(M,    MOD - 2, MOD)     # 1/M      (root filter)

    # ---------- primitive M-th root of unity -----------
    omega = pow(PRIMITIVE_ROOT, (MOD - 1) // M, MOD)

    # ---------- helper powers --------------------------
    two_pow_N = pow(2, N, MOD)

    # ---------- main accumulation over roots -----------
    ans_sum = 0
    lam = 1                                  # ζ^0 == 1
    for _ in range(M):
        c1 = (1 + lam) % MOD
        c2 = (1 - lam) % MOD

        if c1 == 0:                          # λ == -1  (only when M is even)
            # contributes only when n1(u) == N
            s = sum_by_k[N] * two_pow_N % MOD
            ans_sum = (ans_sum + s) % MOD
        else:
            ratio = c2 * pow(c1, MOD - 2, MOD) % MOD      # (1-λ)/(1+λ)
            pref  = pow(c1, N, MOD)                       # (1+λ)^N

            # polynomial evaluation  Σ_{k} sum_by_k[k] * ratio^k
            power = 1
            s = 0
            for coeff in sum_by_k:
                if coeff:
                    s += coeff * power
                power = power * ratio % MOD
            s %= MOD

            ans_sum = (ans_sum + pref * s) % MOD

        lam = lam * omega % MOD            # next root

    # ---------- final answer ---------------------------
    answer = ans_sum * inv_size % MOD
    answer = answer * inv_M % MOD
    print(answer)


if __name__ == "__main__":
    main()