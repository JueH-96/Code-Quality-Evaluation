class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        from itertools import permutations
        
        # Convert each number to its binary representation without the '0b' prefix
        binary_nums = [bin(num)[2:] for num in nums]
        
        # Generate all permutations of the binary representations
        all_permutations = permutations(binary_nums)
        
        # Initialize the maximum binary number
        max_binary_number = 0
        
        # Check each permutation
        for perm in all_permutations:
            # Concatenate the binary strings in the current permutation
            concatenated_bin = ''.join(perm)
            # Convert the concatenated binary string to an integer
            current_number = int(concatenated_bin, 2)
            # Update the maximum binary number found
            max_binary_number = max(max_binary_number, current_number)
        
        return max_binary_number