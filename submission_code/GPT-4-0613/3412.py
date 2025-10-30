class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        return sum(abs(s.index(char) - t.index(char)) for char in s)