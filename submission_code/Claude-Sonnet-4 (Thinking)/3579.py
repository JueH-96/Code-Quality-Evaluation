class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        from itertools import permutations
        
        # Convert each number to binary string (without '0b' prefix)
        binary_strs = [bin(num)[2:] for num in nums]
        
        max_val = 0
        
        # Try all permutations
        for perm in permutations(binary_strs):
            # Concatenate binary strings
            concat_binary = ''.join(perm)
            # Convert back to decimal
            val = int(concat_binary, 2)
            max_val = max(max_val, val)
        
        return max_val