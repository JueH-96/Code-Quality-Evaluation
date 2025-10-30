class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        diffs = []
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                diffs.append(i)
        m = len(diffs)
        if m % 2 != 0:
            return -1
        if m == 0:
            return 0
        diffs.sort()
        total = 0
        for i in range(0, m, 2):
            j = i + 1
            total += min(x, diffs[j] - diffs[i])
        return total