MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    dp = [0] * (N + 1)
    dp[0] = 1  # empty sequence
    
    for i in range(1, N + 1):
        max_k = A[i-1] if i >= 1 else 0
        res = 0
        for k in range(0, max_k + 1):
            if k < A[i-1]:
                res += dp[i-1]
                res %= MOD
            else:
                res += dp[i-1]
                res %= MOD
        dp[i] = res % MOD
    
    print(dp[N] % MOD)

if __name__ == '__main__':
    main()