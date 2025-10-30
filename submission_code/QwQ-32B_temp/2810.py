from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_vals = []
        for j in range(n):
            current_min = float('inf')
            current_list = []
            for k in range(n):
                pos = (j - k) % n
                val = nums[pos]
                if k == 0:
                    current_min = val
                else:
                    current_min = min(current_min, val)
                current_list.append(current_min)
            min_vals.append(current_list)
        
        min_total = float('inf')
        for m in range(n):
            total = m * x
            for j in range(n):
                total += min_vals[j][m]
            if total < min_total:
                min_total = total
        return min_total