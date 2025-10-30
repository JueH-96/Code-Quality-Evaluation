class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        diff = 0
        for i, char_s in enumerate(s):
            for j, char_t in enumerate(t):
                if char_s == char_t:
                    diff += abs(i - j)
                    break
        return diff