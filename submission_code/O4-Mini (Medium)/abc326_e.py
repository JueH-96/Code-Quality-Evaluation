def main():
    import sys
    input = sys.stdin.readline
    MOD = 998244353

    N = int(input())
    A = list(map(int, input().split()))
    
    # Compute modular inverse of N under MOD
    invN = pow(N, MOD - 2, MOD)
    
    # dp_next holds dp[x+1]; we compute dp[x] backwards.
    dp_next = 0
    for x in range(N - 1, -1, -1):
        # s = A[x] + dp_next (mod MOD)
        s = dp_next + A[x]
        if s >= MOD:
            s -= MOD
        # dp[x] = dp_next + s * invN (mod MOD)
        dp_next = dp_next + s * invN % MOD
        if dp_next >= MOD:
            dp_next -= MOD
    
    # dp_next now is dp[0]
    print(dp_next)

if __name__ == "__main__":
    main()