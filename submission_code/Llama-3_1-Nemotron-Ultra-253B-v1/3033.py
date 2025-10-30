class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        diff = []
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                diff.append(i)
        m = len(diff)
        if m % 2 != 0:
            return -1
        cost = 0
        for i in range(0, m, 2):
            d = diff[i+1] - diff[i]
            cost += min(x, d)
        return cost