class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        s_indices = {char: index for index, char in enumerate(s)}
        t_indices = {char: index for index, char in enumerate(t)}
        
        permutation_difference = 0
        for char in s:
            permutation_difference += abs(s_indices[char] - t_indices[char])
        
        return permutation_difference