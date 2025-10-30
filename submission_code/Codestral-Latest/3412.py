class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create a dictionary to store the index of each character in s
        index_map = {char: index for index, char in enumerate(s)}

        # Initialize the permutation difference
        permutation_difference = 0

        # Iterate through each character in t and calculate the absolute difference
        for index, char in enumerate(t):
            permutation_difference += abs(index_map[char] - index)

        return permutation_difference