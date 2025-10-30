class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        from itertools import permutations
        
        max_number = 0
        
        # Generate all permutations of the list nums
        for perm in permutations(nums):
            # Convert each number in the permutation to its binary representation
            binary_strings = [bin(num)[2:] for num in perm]
            # Concatenate the binary strings
            concatenated_binary = ''.join(binary_strings)
            # Convert the concatenated binary string to an integer
            number = int(concatenated_binary, 2)
            # Update max_number if the current number is greater
            max_number = max(max_number, number)
        
        return max_number