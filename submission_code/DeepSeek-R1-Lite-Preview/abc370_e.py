def main():
    import sys
    MOD = 998244353
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    prefix = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix[i] = prefix[i - 1] + A[i - 1]
    
    from collections import defaultdict
    freq = defaultdict(int)
    freq[0] = 1  # S_0 = 0
    
    dp = [0] * (N + 1)
    dp[0] = 1
    
    for i in range(1, N + 1):
        dp[i] = (dp[i - 1] * 2) % MOD
        target = prefix[i] - K
        dp[i] = (dp[i] - freq.get(target, 0)) % MOD
        freq[prefix[i]] = (freq[prefix[i]] + dp[i]) % MOD
    
    print(dp[N] % MOD)

if __name__ == "__main__":
    main()