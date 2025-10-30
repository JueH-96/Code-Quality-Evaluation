from math import sqrt
from itertools import combinations

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def is_perfect_square(n):
            return int(sqrt(n)) ** 2 == n

        max_sum = 0
        for r in range(1, len(nums) + 1):
            for indices in combinations(range(len(nums)), r):
                product = 1
                for i in indices:
                    product *= nums[i]
                if is_perfect_square(product):
                    max_sum = max(max_sum, sum(nums[i] for i in indices))
        return max_sum