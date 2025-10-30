import sys
from collections import defaultdict

def main():
    MOD = 998244353
    N, M, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Initialize DP
    dp = [defaultdict(int) for _ in range(M)]
    dp[0][0] = 1  # Starting with the empty subset
    
    for a in A:
        new_dp = [defaultdict(int) for _ in range(M)]
        for j in range(M):
            for x in list(dp[j].keys()):
                count = dp[j][x]
                # Not taking the current element 'a'
                new_dp[j][x] += count
                # Taking the current element 'a'
                new_j = (j + 1) % M
                new_x = x ^ a
                new_dp[new_j][new_x] += count
        dp = new_dp
    
    total = 0
    for x in dp[0]:
        cnt = dp[0][x]
        if x == 0:
            cnt -= 1  # Subtract the empty subset
        if cnt <= 0:
            continue
        x_pow = pow(x, K, MOD)
        term = (x_pow * cnt) % MOD
        total = (total + term) % MOD
    
    print(total % MOD)

if __name__ == '__main__':
    main()