import math
from collections import defaultdict

MOD = 998244353

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    for m in range(1, n + 1):
        if m == 1:
            print(0)
            continue
        
        # dp[i][g] = number of subsequences ending at position i with last element having GCD g with a[i]
        # sum_dp[i][g] = sum of scores of subsequences ending at position i with last element having GCD g with a[i]
        
        total_score = 0
        
        # For each starting position
        dp = defaultdict(int)
        sum_dp = defaultdict(int)
        
        # Initialize with single element subsequences
        dp[a[0]] = 1
        sum_dp[a[0]] = 0
        
        for i in range(1, m):
            new_dp = defaultdict(int)
            new_sum_dp = defaultdict(int)
            
            # Single element subsequence starting at position i
            new_dp[a[i]] = 1
            new_sum_dp[a[i]] = 0
            
            # Extend previous subsequences
            for prev_val, count in dp.items():
                g = math.gcd(prev_val, a[i])
                new_dp[a[i]] = (new_dp[a[i]] + count) % MOD
                new_sum_dp[a[i]] = (new_sum_dp[a[i]] + sum_dp[prev_val] + g * count) % MOD
            
            # Update totals
            for val in new_sum_dp:
                total_score = (total_score + new_sum_dp[val]) % MOD
            
            # Merge new_dp into dp for next iteration
            for val, count in new_dp.items():
                dp[val] = (dp[val] + count) % MOD
                sum_dp[val] = (sum_dp[val] + new_sum_dp[val]) % MOD
        
        print(total_score)

solve()