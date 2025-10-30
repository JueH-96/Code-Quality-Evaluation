import bisect
from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + nums[i - 1]
        
        dp_list = [(0, 0, 0)]  # Each entry is (x_j, max_groups, s_j)
        x_values = [0]  # Track x_j values for bisect
        
        max_groups = 0
        
        for i in range(1, n + 1):
            s_i = prefix_sum[i]
            # Find the largest index where x_j <= s_i
            idx = bisect.bisect_right(x_values, s_i) - 1
            if idx < 0:
                continue  # Should not happen as x=0 is always present
            
            x_j, current_max, s_j_prev = dp_list[idx]
            new_max = current_max + 1
            new_last_sum = s_i - s_j_prev
            new_x = s_i + new_last_sum
            
            # Check if we should add this new entry to dp_list
            if new_x >= x_values[-1]:
                if new_max > dp_list[-1][1]:
                    dp_list.append((new_x, new_max, s_i))
                    x_values.append(new_x)
            
            if new_max > max_groups:
                max_groups = new_max
        
        return max_groups