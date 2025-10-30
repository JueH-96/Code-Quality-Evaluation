class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        max_num = max(nums)
        sum_diffs = 0
        max_diff = 0
        for num in nums:
            diff = max_num - num
            sum_diffs += diff
            if diff > max_diff:
                max_diff = diff
        if cost2 < 2 * cost1:
            p = min(sum_diffs // 2, sum_diffs - max_diff)
            cost = p * cost2 + (sum_diffs - 2 * p) * cost1
        else:
            cost = sum_diffs * cost1
        return cost % MOD