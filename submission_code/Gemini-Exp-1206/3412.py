class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        s_indices = {}
        t_indices = {}
        for i, char in enumerate(s):
            s_indices[char] = i
        for i, char in enumerate(t):
            t_indices[char] = i
        permutation_difference = 0
        for char in s_indices:
            permutation_difference += abs(s_indices[char] - t_indices[char])
        return permutation_difference