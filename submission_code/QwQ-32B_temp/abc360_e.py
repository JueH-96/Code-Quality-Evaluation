MOD = 998244353

def main():
    N, K = map(int, input().split())
    a = N - 2
    pow_a = pow(a, K, MOD)
    pow_b = pow(N, K, MOD)
    inv_b = pow(pow_b, MOD-2, MOD)
    T = (pow_a * inv_b) % MOD
    term1 = (N + 1) % MOD
    term2 = (N - 1) % MOD
    numerator = (term1 - term2 * T) % MOD
    inv2 = pow(2, MOD-2, MOD)
    ans = (numerator * inv2) % MOD
    print(ans)

if __name__ == "__main__":
    main()