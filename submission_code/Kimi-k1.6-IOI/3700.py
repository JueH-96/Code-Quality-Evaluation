from collections import Counter
from typing import List

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        result = 0
        
        def comb(n, k):
            if k < 0 or k > n:
                return 0
            if k == 0:
                return 1
            if k == 1:
                return n
            if k == 2:
                return n * (n - 1) // 2
            return 0  # This case won't be used in our problem constraints
        
        for i in range(n):
            x = nums[i]
            left = nums[:i]
            right = nums[i+1:]
            
            left_x = left.count(x)
            right_x = right.count(x)
            
            left_counter = Counter(left)
            right_counter = Counter(right)
            
            left_non_x = {k: v for k, v in left_counter.items() if k != x}
            right_non_x = {k: v for k, v in right_counter.items() if k != x}
            
            total_left_non_x = sum(left_non_x.values())
            total_right_non_x = sum(right_non_x.values())
            
            for a in [0, 1]:
                b = 1 - a
                if a > left_x or b > right_x:
                    continue
                if a < 0 or b < 0:
                    continue
                
                if (a + b) >= 2:
                    left_ways = comb(left_x, a) * comb(total_left_non_x, 2 - a)
                    right_ways = comb(right_x, b) * comb(total_right_non_x, 2 - b)
                    result = (result + left_ways * right_ways) % MOD
                else:
                    left_values = set(left_non_x.keys())
                    right_values = set(right_non_x.keys())
                    common = left_values & right_values
                    left_only = left_values - common
                    right_only = right_values - common
                    
                    left_only_counts = [left_non_x[v] for v in left_only]
                    right_only_counts = [right_non_x[v] for v in right_only]
                    common_counts = [(left_non_x[v], right_non_x[v]) for v in common]
                    
                    dp = [[0] * 3 for _ in range(3)]
                    dp[0][0] = 1
                    
                    for cnt in left_only_counts:
                        for i in range(2, -1, -1):
                            for j in range(2, -1, -1):
                                if dp[i][j]:
                                    if i + 1 <= 2:
                                        dp[i+1][j] = (dp[i+1][j] + dp[i][j] * cnt) % MOD
                    
                    for cnt in right_only_counts:
                        for i in range(2, -1, -1):
                            for j in range(2, -1, -1):
                                if dp[i][j]:
                                    if j + 1 <= 2:
                                        dp[i][j+1] = (dp[i][j+1] + dp[i][j] * cnt) % MOD
                    
                    for a_v, b_v in common_counts:
                        for i in range(2, -1, -1):
                            for j in range(2, -1, -1):
                                current = dp[i][j]
                                if current == 0:
                                    continue
                                if i + 1 <= 2:
                                    dp[i+1][j] = (dp[i+1][j] + current * a_v) % MOD
                                if j + 1 <= 2:
                                    dp[i][j+1] = (dp[i][j+1] + current * b_v) % MOD
                                if i + 1 <= 2 and j + 1 <= 2:
                                    dp[i+1][j+1] = (dp[i+1][j+1] - current * a_v * b_v) % MOD
                                    dp[i+1][j+1] = (dp[i+1][j+1] + MOD) % MOD  # Ensure non-negative
                    
                    required_x = 2 - a
                    required_y = 1 + a
                    valid_pairs = dp[required_x][required_y] if required_x <= 2 and required_y <= 2 else 0
                    
                    left_ways = comb(left_x, a)
                    right_ways = comb(right_x, b)
                    add = (left_ways * right_ways) % MOD
                    add = (add * valid_pairs) % MOD
                    result = (result + add) % MOD
        
        return result % MOD