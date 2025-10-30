from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7

        # If cost2 is more than twice cost1, it's always better to use cost1
        if cost2 > 2 * cost1:
            min_val = min(nums)
            return sum((x - min_val) * cost1 for x in nums) % MOD

        # Otherwise, we need to consider both operations
        max_val = max(nums)
        min_cost = float('inf')

        for target in range(min(nums), max_val + 1):
            cost = 0
            freq = {}
            for num in nums:
                if num in freq:
                    freq[num] += 1
                else:
                    freq[num] = 1

            for num, count in freq.items():
                if num < target:
                    diff = target - num
                    pairs = count // 2
                    singles = count % 2
                    cost += pairs * cost2 + singles * cost1
                    cost += diff * (pairs * 2 + singles) * cost1
                elif num > target:
                    diff = num - target
                    pairs = count // 2
                    singles = count % 2
                    cost += pairs * cost2 + singles * cost1
                    cost += diff * (pairs * 2 + singles) * cost1

            min_cost = min(min_cost, cost)

        return min_cost % MOD