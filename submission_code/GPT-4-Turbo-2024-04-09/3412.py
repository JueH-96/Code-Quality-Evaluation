class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create a dictionary to store the index of each character in t
        index_in_t = {char: idx for idx, char in enumerate(t)}
        
        # Calculate the permutation difference
        permutation_difference = 0
        for idx, char in enumerate(s):
            permutation_difference += abs(idx - index_in_t[char])
        
        return permutation_difference