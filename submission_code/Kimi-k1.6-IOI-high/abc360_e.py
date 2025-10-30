MOD = 998244353

def main():
    import sys
    N, K = map(int, sys.stdin.readline().split())
    
    a = pow(N, K, MOD)
    b = pow(N-2, K, MOD) if N != 1 else 0  # Handle N=1 case (though not necessary as pow(-1, K, MOD) works)
    
    term1 = (N + 1) * a % MOD
    term2 = (N - 1) * b % MOD
    numerator = (term1 - term2) % MOD
    denominator = (2 * a) % MOD
    inv_denominator = pow(denominator, MOD-2, MOD)
    
    ans = numerator * inv_denominator % MOD
    print(ans)

if __name__ == "__main__":
    main()