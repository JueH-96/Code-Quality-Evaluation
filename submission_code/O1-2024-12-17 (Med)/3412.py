class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Map each character in s to its index
        char_index = {char: i for i, char in enumerate(s)}

        total_difference = 0
        # Sum the absolute differences of indices for each character
        for i, char in enumerate(t):
            total_difference += abs(char_index[char] - i)

        return total_difference