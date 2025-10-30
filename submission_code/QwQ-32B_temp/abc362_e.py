import sys
from collections import defaultdict

MOD = 998244353

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    
    dp = [defaultdict(lambda: defaultdict(int)) for _ in range(N)]
    
    for i in range(N):
        for j in range(i):
            d = A[i] - A[j]
            # Handle the pair (j, i)
            dp[i][d][2] += 1
            dp[i][d][2] %= MOD
            # Check existing subsequences in dp[j][d]
            if d in dp[j]:
                for l in dp[j][d]:
                    cnt = dp[j][d][l]
                    new_l = l + 1
                    dp[i][d][new_l] = (dp[i][d][new_l] + cnt) % MOD
    
    ans = [0] * (N + 1)
    ans[1] = N
    for k in range(2, N + 1):
        total = 0
        for i in range(N):
            for d in dp[i]:
                if k in dp[i][d]:
                    total = (total + dp[i][d][k]) % MOD
        ans[k] = total
    
    print(' '.join(map(str, ans[1:N+1])))

if __name__ == "__main__":
    main()