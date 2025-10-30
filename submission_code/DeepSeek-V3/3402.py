class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        max_num = max(nums)
        n = len(nums)
        total = 0
        for num in nums:
            total += max_num - num
        if cost1 * 2 <= cost2:
            return (total * cost1) % MOD
        else:
            # Calculate the number of pairs we can form
            pairs = total // 2
            remaining = total % 2
            cost = pairs * cost2 + remaining * cost1
            return cost % MOD