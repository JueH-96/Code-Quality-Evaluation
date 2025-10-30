class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create a dictionary to map each character to its index in s
        s_index = {char: idx for idx, char in enumerate(s)}
        # Create a dictionary to map each character to its index in t
        t_index = {char: idx for idx, char in enumerate(t)}
        # Calculate the sum of absolute differences
        difference = 0
        for char in s:
            difference += abs(s_index[char] - t_index[char])
        return difference