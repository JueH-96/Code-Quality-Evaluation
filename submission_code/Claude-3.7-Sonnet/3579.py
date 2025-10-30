class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        from itertools import permutations
        
        # Convert integers to their binary representations without leading zeros
        binary_strings = [bin(num)[2:] for num in nums]
        
        max_decimal = 0
        
        # Try all possible permutations of the binary strings
        for perm in permutations(binary_strings):
            # Concatenate the binary strings in this order
            concatenated = ''.join(perm)
            # Convert the concatenated binary string to decimal
            decimal = int(concatenated, 2)
            # Update the maximum value
            max_decimal = max(max_decimal, decimal)
        
        return max_decimal