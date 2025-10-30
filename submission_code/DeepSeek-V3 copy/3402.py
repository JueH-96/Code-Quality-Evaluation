class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        max_num = max(nums)
        n = len(nums)
        total = 0
        for num in nums:
            total += max_num - num
        if cost2 >= 2 * cost1:
            return (total * cost1) % MOD
        else:
            # Calculate the number of pairs we can form
            pairs = total // 2
            # Calculate the remaining single increments
            remainder = total % 2
            # Cost is pairs * cost2 + remainder * cost1
            cost = (pairs * cost2 + remainder * cost1) % MOD
            return cost