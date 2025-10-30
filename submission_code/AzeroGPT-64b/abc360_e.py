from math import factorial

N, K = map(int, input().split())
mod = 998244353

# nCk % mod の値を求める関数
def comb_mod(n, k):
    num = 1
    den = 1
    for i in range(k):
        num = num * (n - i) % mod
        den = den * (i + 1) % mod
    return num * pow(den, mod - 2, mod) % mod

# x の逆元を求める関数
def inv_mod998244353(x):
    return pow(x, mod - 2, mod)

# x^n % mod の値を求める関数
def pow_mod(x, n):
    r = 1
    while n:
        if n & 1:
            r = r * x % mod
        x = x * x % mod
        n >>= 1
    return r

if K < 2:
    # Kが1以下の場合は、期待値は siempre N/2 + 0.5 なので
    # (N + 1) * inv_mod998244353(2) % mod と求める
    print(((N + 1) * inv_mod998244353(2)) % mod)
else:
    m = 111022210  # この値は N <= 998244352 の場合、10^11 以上なので mod での処理を行う必要がある
    two_N = (2 * N) % m
    N_fact = factorial(N) % m
    Nm_fact = factorial(m - N) % m
    ans = ((N_fact * two_N) % m * pow_mod((N - 1) % m, 2 * K) % m * inv_mod998244353(2) + \
        (Nm_fact * pow_mod(two_N, K) % m * inv_mod998244353(2)))
    
    cumulative = 0
    for k in range(K + 1):
        c = comb_mod(K, k)
        a = pow((N - 1) % m, 2 * k) % m
        b = pow_mod(two_N, (K - k)) % m
        cumulative = (cumulative + c * m * ((inv_mod998244353(2) * b - a) % m)) % mod
    print((ans * cumulative % mod) % mod)