class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        n = len(s)
        diff = []
        for i in range(n):
            if s[i] != t[i]:
                diff.append(i)
        if len(diff) == 0:
            return 0
        if len(diff) == 1:
            return abs(diff[0] - (n - diff[0]))
        min_diff = abs(diff[0] - diff[1])
        for i in range(1, len(diff) - 1):
            min_diff = min(min_diff, abs(diff[i] - diff[i + 1]))
        return min_diff