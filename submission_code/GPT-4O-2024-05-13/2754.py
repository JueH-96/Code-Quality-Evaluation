class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        from itertools import combinations
        
        n = len(nums)
        max_strength = float('-inf')
        
        # Generate all non-empty subsets
        for r in range(1, n + 1):
            for subset in combinations(nums, r):
                product = 1
                for num in subset:
                    product *= num
                max_strength = max(max_strength, product)
        
        return max_strength