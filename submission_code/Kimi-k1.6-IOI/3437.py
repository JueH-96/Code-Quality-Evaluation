import bisect
from typing import List
from collections import defaultdict

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        sum_x = defaultdict(int)
        for num in power:
            sum_x[num] += num
        
        unique_x = sorted(sum_x.keys())
        m = len(unique_x)
        if m == 0:
            return 0
        
        sum_x_list = [sum_x[x] for x in unique_x]
        x_list = unique_x
        
        dp = [0] * m
        dp[0] = sum_x_list[0]
        max_sum = dp[0]
        
        for i in range(1, m):
            x = x_list[i]
            target = x - 3
            j = bisect.bisect_right(x_list, target) - 1
            if j >= 0:
                current_sum = sum_x_list[i] + dp[j]
            else:
                current_sum = sum_x_list[i]
            dp[i] = max(current_sum, dp[i-1])
            if dp[i] > max_sum:
                max_sum = dp[i]
        
        return max_sum