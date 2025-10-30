class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        max_num = max(nums)
        min_num = min(nums)
        
        def calculate_cost(target):
            total_cost = 0
            for num in nums:
                diff = target - num
                if cost1 * 2 <= cost2:
                    total_cost += diff * cost1
                else:
                    total_cost += (diff // 2) * cost2 + (diff % 2) * cost1
            return total_cost
        
        min_cost = float('inf')
        for target in range(min_num, max_num + 1):
            min_cost = min(min_cost, calculate_cost(target))
        
        return min_cost % MOD