class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        diff = 0
        for char in s:
            diff += abs(s.index(char) - t.index(char))
        return diff