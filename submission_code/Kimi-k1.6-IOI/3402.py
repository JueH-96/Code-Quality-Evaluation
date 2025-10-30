class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        if not nums:
            return 0
        max_num = max(nums)
        sum_deltas = 0
        max_delta = 0
        for num in nums:
            delta = max_num - num
            sum_deltas += delta
            if delta > max_delta:
                max_delta = delta
        # Calculate the maximum possible two-increment operations
        k_max = min(sum_deltas // 2, sum_deltas - max_delta)
        if cost2 < 2 * cost1:
            total = (k_max * cost2 + (sum_deltas - 2 * k_max) * cost1) % MOD
        else:
            total = (sum_deltas * cost1) % MOD
        return total