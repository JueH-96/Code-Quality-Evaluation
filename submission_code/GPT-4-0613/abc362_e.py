import sys
from collections import defaultdict
MOD = 998244353

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    dp = [[0]*81 for _ in range(81)]
    dp[0][0] = 1
    cnt = defaultdict(int)
    for a in A:
        for i in reversed(range(81)):
            dp[a][i] += dp[a][i-1]
            dp[a][i] %= MOD
        for b in range(81):
            if a == b:
                continue
            dp[a][1] += dp[b][0]
            dp[a][1] %= MOD
        cnt[a] += 1
    ans = [0]*81
    for a in range(81):
        for i in range(81):
            ans[i] += dp[a][i]
            ans[i] %= MOD
    print(*ans[1:N+1])

if __name__ == "__main__":
    main()