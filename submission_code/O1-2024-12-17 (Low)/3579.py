class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        from itertools import permutations
        
        max_val = 0
        for perm in permutations(nums):
            # Build the concatenated binary string for this permutation
            bin_str = ''.join(bin(num)[2:] for num in perm)
            # Convert this binary string to an integer
            val = int(bin_str, 2)
            # Keep track of the maximum value
            max_val = max(max_val, val)
        
        return max_val