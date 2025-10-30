from itertools import combinations

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        max_strength = -float('inf')
        n = len(nums)
        # Iterate over all possible non-empty subsets
        for r in range(1, n+1):
            for subset in combinations(nums, r):
                product = 1
                for num in subset:
                    product *= num
                if product > max_strength:
                    max_strength = product
        return max_strength