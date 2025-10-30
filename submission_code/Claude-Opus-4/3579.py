class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        from itertools import permutations
        
        max_val = 0
        
        # Try all permutations of the 3 numbers
        for perm in permutations(nums):
            # Concatenate binary representations (without '0b' prefix)
            binary_str = ''
            for num in perm:
                binary_str += bin(num)[2:]
            
            # Convert concatenated binary string to decimal
            decimal_val = int(binary_str, 2)
            max_val = max(max_val, decimal_val)
        
        return max_val