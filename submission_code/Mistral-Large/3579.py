from typing import List

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        # Convert each number to its binary representation without the '0b' prefix
        bin_repr = [bin(num)[2:] for num in nums]

        # Sort the binary representations by their length in descending order
        # If lengths are equal, sort lexicographically in descending order
        bin_repr.sort(key=lambda x: (len(x), x), reverse=True)

        # Concatenate the sorted binary representations
        concatenated_bin = ''.join(bin_repr)

        # Convert the concatenated binary string back to an integer
        return int(concatenated_bin, 2)