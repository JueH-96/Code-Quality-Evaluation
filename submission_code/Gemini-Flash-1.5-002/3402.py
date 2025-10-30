class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        n = len(nums)
        mod = 10**9 + 7

        min_val = min(nums)
        max_val = max(nums)

        min_cost = float('inf')

        for target in range(min_val, max_val + 1):
            cost = 0
            increase_count = 0
            decrease_count = 0
            for num in nums:
                diff = target - num
                if diff > 0:
                    increase_count += diff
                elif diff < 0:
                    decrease_count += -diff

            if cost1 < cost2 * 2:
                cost = increase_count * cost1 + decrease_count * cost1
            else:
                cost = min(increase_count, decrease_count) * cost2 + abs(increase_count - decrease_count) * cost1

            min_cost = min(min_cost, cost)

        return min_cost