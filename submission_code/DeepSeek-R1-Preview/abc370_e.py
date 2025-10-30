import sys
from collections import defaultdict

MOD = 998244353

def main():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Compute prefix sums
    S = [0] * (N + 1)
    for i in range(1, N + 1):
        S[i] = S[i-1] + A[i-1]
    
    sum_total = 1  # Initially, sum_total is dp[0] = 1
    cnt = defaultdict(int)
    cnt[S[0]] = 1  # S[0] is 0, and dp[0] is 1
    
    for i in range(1, N + 1):
        s_i = S[i]
        target = s_i - K
        sum_bad = cnt.get(target, 0)
        dp_i = (sum_total - sum_bad) % MOD
        sum_total = (sum_total + dp_i) % MOD
        cnt[s_i] = (cnt[s_i] + dp_i) % MOD
    
    print(dp_i % MOD)

if __name__ == "__main__":
    main()