import bisect
from typing import List

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        
        # Generate all unique differences
        differences = set()
        for i in range(n):
            for j in range(i + 1, n):
                diff = sorted_nums[j] - sorted_nums[i]
                differences.add(diff)
        differences = sorted(differences)
        differences.append(float('inf'))  # Add a dummy value at the end
        
        # Function to compute f(d)
        def compute_f(d):
            dp = [[0] * (k + 1) for _ in range(n)]
            prefix_sum = [[0] * n for _ in range(k + 1)]
            
            # Initialize for j=1
            for i in range(n):
                dp[i][1] = 1 % MOD
                if i == 0:
                    prefix_sum[1][i] = dp[i][1]
                else:
                    prefix_sum[1][i] = (prefix_sum[1][i - 1] + dp[i][1]) % MOD
            
            # Fill for j from 2 to k
            for j in range(2, k + 1):
                for i in range(n):
                    target = sorted_nums[i] - d
                    if i == 0:
                        m = -1
                    else:
                        m = bisect.bisect_right(sorted_nums, target, 0, i) - 1
                    sum_val = 0
                    if m >= 0:
                        if j - 1 <= k and m < n:
                            sum_val = prefix_sum[j - 1][m]
                    sum_val %= MOD
                    dp[i][j] = sum_val
                    # Update prefix_sum[j][i]
                    if i == 0:
                        prefix_sum[j][i] = dp[i][j] % MOD
                    else:
                        prefix_sum[j][i] = (prefix_sum[j][i - 1] + dp[i][j]) % MOD
            
            # Compute total f(d)
            total = 0
            for i in range(n):
                total = (total + dp[i][k]) % MOD
            return total
        
        sum_powers = 0
        m = len(differences)
        for i in range(m - 1):
            d = differences[i]
            next_d = differences[i + 1]
            f_d = compute_f(d)
            f_next = compute_f(next_d)
            contribution = (d * ((f_d - f_next) % MOD)) % MOD
            sum_powers = (sum_powers + contribution) % MOD
        
        return sum_powers % MOD