class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        diffs = []
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                diffs.append(i)
        if len(diffs) % 2 != 0:
            return -1
        diffs.sort()
        cost = 0
        for i in range(0, len(diffs), 2):
            j = i + 1
            if j >= len(diffs):
                break
            distance = diffs[j] - diffs[i]
            cost += min(x, distance)
        return cost