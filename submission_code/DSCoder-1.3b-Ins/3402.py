class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        mod = 10**9 + 7
        min_num = min(nums)
        total_cost = sum(num - min_num for num in nums) * cost1
        total_cost += sum(num - min_num for num in nums) * cost2
        return total_cost % mod