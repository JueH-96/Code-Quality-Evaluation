import sys
from collections import defaultdict

def main():
    MOD = 998244353
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    
    dp = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]
    
    for i in range(n):
        for j in range(i):
            d = a[i] - a[j]
            # Add the subsequence of length 2
            dp[i][d][2] = (dp[i][d][2] + 1) % MOD
            # Extend existing subsequences
            curr = dp[j][d]
            for m in curr:
                new_m = m + 1
                dp[i][d][new_m] = (dp[i][d][new_m] + curr[m]) % MOD
    
    ans = [0] * (n + 1)
    ans[1] = n % MOD
    
    for k in range(2, n + 1):
        res = 0
        for i in range(n):
            for d in dp[i]:
                res = (res + dp[i][d].get(k, 0)) % MOD
        ans[k] = res
    
    output = [str(ans[i]) for i in range(1, n + 1)]
    print(' '.join(output))

if __name__ == "__main__":
    main()