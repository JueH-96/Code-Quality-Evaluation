class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create dictionaries to store the index of each character in s and t
        s_index = {char: idx for idx, char in enumerate(s)}
        t_index = {char: idx for idx, char in enumerate(t)}
        
        # Calculate the permutation difference
        permutation_difference = 0
        for char in s:
            permutation_difference += abs(s_index[char] - t_index[char])
        
        return permutation_difference