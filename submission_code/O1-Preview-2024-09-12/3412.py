class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        positions_s = {}
        positions_t = {}
        for index, c in enumerate(s):
            positions_s[c] = index
        for index, c in enumerate(t):
            positions_t[c] = index
        total_difference = 0
        for c in positions_s:
            total_difference += abs(positions_s[c] - positions_t[c])
        return total_difference