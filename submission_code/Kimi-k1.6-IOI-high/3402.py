class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        max_num = max(nums)
        d = [max_num - num for num in nums]
        sum_d = sum(d)
        if sum_d == 0:
            return 0
        max_d = max(d)
        x = 0
        if cost2 < 2 * cost1:
            x = min(sum_d // 2, sum_d - max_d)
        total_cost = (sum_d * cost1 - x * (2 * cost1 - cost2)) % MOD
        return total_cost