class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        permutation_difference = 0
        for char_s in s:
            index_s = s.index(char_s)
            index_t = t.index(char_s)
            permutation_difference += abs(index_s - index_t)
        return permutation_difference