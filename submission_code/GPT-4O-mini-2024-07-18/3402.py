class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        from bisect import bisect_left
        from collections import Counter
        
        MOD = 10**9 + 7
        
        def calculate_cost(target):
            cost = 0
            for num in nums:
                diff = target - num
                if diff > 0:
                    if diff % 2 == 0:
                        cost += (diff // 2) * cost2 + (diff // 2) * cost1
                    else:
                        cost += (diff // 2) * cost2 + ((diff // 2) + 1) * cost1
                cost %= MOD
            return cost
        
        min_cost = float('inf')
        unique_nums = sorted(set(nums))
        
        for target in unique_nums:
            min_cost = min(min_cost, calculate_cost(target))
        
        return min_cost