class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        total_cost = 0
        nums.sort()
        max_val = nums[-1]

        diffs = []
        for num in nums:
            diffs.append(max_val - num)

        if 2 * cost1 <= cost2:
            for diff in diffs:
                total_cost = (total_cost + diff * cost1) % MOD
        else:
            total_diff = sum(diffs)
            total_cost = (total_diff // 2) * cost2 + (total_diff % 2) * cost1
            total_cost %= MOD

        return total_cost