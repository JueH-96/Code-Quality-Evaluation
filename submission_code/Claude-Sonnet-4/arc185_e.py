import math

def solve():
    MOD = 998244353
    
    n = int(input())
    A = list(map(int, input().split()))
    
    for m in range(1, n + 1):
        # dp[val] = [count, sum] where count is number of subsequences ending with val,
        # and sum is the total score of all such subsequences
        dp = {}
        
        for i in range(m):
            val = A[i]
            new_dp = dp.copy()
            
            # Single element subsequence
            if val not in new_dp:
                new_dp[val] = [0, 0]
            new_dp[val][0] = (new_dp[val][0] + 1) % MOD
            
            # Extend existing subsequences
            for prev_val in dp:
                if dp[prev_val][0] > 0:
                    gcd_val = math.gcd(prev_val, val)
                    count = dp[prev_val][0]
                    prev_sum = dp[prev_val][1]
                    
                    if val not in new_dp:
                        new_dp[val] = [0, 0]
                    new_dp[val][0] = (new_dp[val][0] + count) % MOD
                    new_dp[val][1] = (new_dp[val][1] + prev_sum + count * gcd_val) % MOD
            
            dp = new_dp
        
        total_score = sum(dp[val][1] for val in dp) % MOD
        print(total_score)

solve()