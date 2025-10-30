import sys

MOD = 998244353

# ------------------------------------------------------------
# factorials and combinations  ( up to N = W+H+4 <= 2*10^6+4 )
# ------------------------------------------------------------
def build_fact_inv(N):
    fact = [1]*(N+1)
    for i in range(1, N+1):
        fact[i] = fact[i-1]*i % MOD
    invfact = [1]*(N+1)
    invfact[N] = pow(fact[N], MOD-2, MOD)
    for i in range(N, 0, -1):
        invfact[i-1] = invfact[i]*i % MOD
    return fact, invfact

def C(n, k, fact, invfact):
    if k < 0 or k > n:           # we will need that quite often
        return 0
    return fact[n] * invfact[k] % MOD * invfact[n-k] % MOD

# ------------------------------------------------------------
#  dp_full   and   T  (formulas (1) and (3))
# ------------------------------------------------------------
def dp_full(a, b, fact, invfact):           # a = W-x , b = H-y
    return (C(a+b+2, a+1, fact, invfact) - 1) % MOD

def T_prefix(x, y, fact, invfact):          # x , y  themselves
    return (C(x+y+2, x+1, fact, invfact) - 1) % MOD

# ------------------------------------------------------------
def main() -> None:
    W, H, L, R, D, U = map(int, sys.stdin.readline().split())

    Nmax = W + H + 4
    fact, invfact = build_fact_inv(Nmax)

    # --------------------------------------------------------
    # 1. ALL  (formula (2))
    # --------------------------------------------------------
    ALL = (C(W+H+4, W+2, fact, invfact) - (W+H+4) - (W+1)*(H+1)) % MOD

    # --------------------------------------------------------
    # 2. entering through west / south border
    # --------------------------------------------------------
    bad = 0

    if L > 0:                            # west border, x = L
        x_out = L-1
        for y in range(D, U+1):
            pref = T_prefix(x_out, y, fact, invfact)
            suf  = dp_full(W-L, H-y, fact, invfact)   # W-x , H-y
            bad = (bad + pref * suf) % MOD

    if D > 0:                            # south border, y = D
        y_out = D-1
        for x in range(L, R+1):
            pref = T_prefix(x, y_out, fact, invfact)
            suf  = dp_full(W-x, H-D, fact, invfact)
            bad = (bad + pref * suf) % MOD

    # --------------------------------------------------------
    # 3. starts inside the hole
    # --------------------------------------------------------
    Amin, Amax = W - R, W - L           # a = W-x
    Bmin, Bmax = H - U, H - D           # b = H-y

    binom_rect = 0
    for a in range(Amin, Amax+1):
        big  = C(a+Bmax+3, a+2, fact, invfact)
        small = C(a+Bmin+2, a+2, fact, invfact)
        binom_rect = (binom_rect + big - small) % MOD

    blocks_hole = (R-L+1)*(U-D+1) % MOD
    start_in_hole = (binom_rect - blocks_hole) % MOD

    # --------------------------------------------------------
    ans = (ALL - bad - start_in_hole) % MOD
    print(ans)

if __name__ == "__main__":
    main()