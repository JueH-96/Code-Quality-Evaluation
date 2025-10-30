from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        min_val = min(nums)
        ans = float('inf')
        
        for target in range(min_val, max(nums) + 1):
            cost = 0
            diffs = [target - num for num in nums if num < target]
            
            total_diff = sum(diffs)
            
            if cost1 <= cost2 * 2:
                cost = total_diff * cost1
            else:
                cost = (total_diff // 2) * cost2 + (total_diff % 2) * cost1
            
            ans = min(ans, cost)
        
        return ans % (10**9 + 7)