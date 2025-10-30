class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        permutation_difference = 0
        for i in range(len(s)):
            char_s = s[i]
            index_s = i
            index_t = t.find(char_s)
            permutation_difference += abs(index_s - index_t)
        return permutation_difference