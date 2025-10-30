class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create dictionaries to store the index of each character in s and t
        index_s = {char: idx for idx, char in enumerate(s)}
        index_t = {char: idx for idx, char in enumerate(t)}

        # Calculate the permutation difference
        permutation_difference = sum(abs(index_s[char] - index_t[char]) for char in s)

        return permutation_difference