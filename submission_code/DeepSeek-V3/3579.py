from itertools import permutations

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_num = 0
        # Generate all possible permutations of the nums array
        for p in permutations(nums):
            # Concatenate the binary representations of the numbers in the current permutation
            binary_str = ''
            for num in p:
                binary_str += bin(num)[2:]  # bin(num) returns a string like '0b1010', so we slice off the '0b'
            # Convert the binary string to a decimal number
            current_num = int(binary_str, 2)
            # Update max_num if the current number is greater
            if current_num > max_num:
                max_num = current_num
        return max_num