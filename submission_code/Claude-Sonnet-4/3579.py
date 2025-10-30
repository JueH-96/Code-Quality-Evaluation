class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        from itertools import permutations
        
        # Convert numbers to binary strings (without '0b' prefix)
        binary_strs = [bin(num)[2:] for num in nums]
        
        max_value = 0
        
        # Try all permutations of the binary strings
        for perm in permutations(binary_strs):
            # Concatenate the binary strings
            concatenated = ''.join(perm)
            # Convert back to decimal
            decimal_value = int(concatenated, 2)
            # Track maximum
            max_value = max(max_value, decimal_value)
        
        return max_value