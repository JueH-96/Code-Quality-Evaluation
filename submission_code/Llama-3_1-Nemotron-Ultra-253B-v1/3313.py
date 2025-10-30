from typing import List
import sys
import math

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        dp = [[-math.inf] * n for _ in range(k + 1)]
        
        for j in range(1, k + 1):
            c = (-1) ** (j + 1) * (k - j + 1)
            current_dp = [-math.inf] * n
            running_max = -math.inf
            
            for i in range(n):
                s = i
                if s == 0:
                    if j - 1 == 0:
                        val = 0 - c * prefix[0]
                    else:
                        val = -math.inf
                else:
                    if j - 1 >= 1:
                        val = dp[j - 1][s - 1] - c * prefix[s]
                    else:
                        val = -math.inf
                
                running_max = max(running_max, val)
                current_term = c * prefix[i + 1] + running_max
                
                if i == 0:
                    current_dp[i] = current_term
                else:
                    current_dp[i] = max(current_dp[i - 1], current_term)
            
            dp[j] = current_dp
        
        max_strength = max(dp[k])
        return max_strength if max_strength != -math.inf else 0