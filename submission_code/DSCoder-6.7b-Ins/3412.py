class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        difference = 0
        for i in range(len(s)):
            difference += abs(s.index(s[i]) - t.index(s[i]))
        return difference