class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        total = 0
        for i, char in enumerate(s):
            j = t.index(char)
            total += abs(i - j)
        return total