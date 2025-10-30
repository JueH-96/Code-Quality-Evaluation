class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create a dictionary to store the index of each character in s
        index_map = {char: idx for idx, char in enumerate(s)}
        # Calculate the permutation difference
        permutation_difference = sum(abs(index_map[char] - idx) for idx, char in enumerate(t))
        return permutation_difference