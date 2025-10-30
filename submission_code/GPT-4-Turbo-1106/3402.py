class Solution:
    def minCostToEqualizeArray(self, nums: list[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        max_num = max(nums)
        min_cost = float('inf')
        
        for target in range(max_num, max_num - len(nums), -1):
            single_ops = 0
            double_ops = 0
            for num in nums:
                diff = target - num
                double_ops += diff // 2
                single_ops += diff % 2
            if double_ops <= single_ops:
                cost = double_ops * cost2 + (single_ops - double_ops) * cost1
            else:
                cost = single_ops * cost2 + (double_ops - single_ops) // 2 * cost2
                if (double_ops - single_ops) % 2:
                    cost += cost1
            min_cost = min(min_cost, cost)
        
        return min_cost % MOD