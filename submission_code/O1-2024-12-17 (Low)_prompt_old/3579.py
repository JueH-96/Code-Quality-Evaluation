class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        from itertools import permutations
        
        max_value = 0
        
        # Generate all permutations of the three numbers
        for combo in permutations(nums):
            # Build the binary string by concatenating each integer's binary representation (without the '0b' prefix)
            binary_str = ''.join(bin(num)[2:] for num in combo)
            # Convert the binary string back to an integer
            current_value = int(binary_str, 2)
            
            # Update the maximum if current_value is larger
            max_value = max(max_value, current_value)
        
        return max_value