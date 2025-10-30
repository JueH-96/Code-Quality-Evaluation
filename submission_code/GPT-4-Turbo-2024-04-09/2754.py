class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        from functools import reduce
        from itertools import combinations
        
        def product_of_list(lst):
            return reduce(lambda x, y: x * y, lst, 1)
        
        n = len(nums)
        max_strength = float('-inf')
        
        # Check all possible non-empty subsets
        for r in range(1, n + 1):
            for subset in combinations(nums, r):
                current_strength = product_of_list(subset)
                if current_strength > max_strength:
                    max_strength = current_strength
        
        return max_strength