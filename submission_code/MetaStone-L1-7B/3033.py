class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        diff = []
        for i in range(n):
            if s1[i] != s2[i]:
                diff.append(i)
        d = len(diff)
        if d % 2 != 0:
            return -1
        
        sorted_diff = sorted(diff)
        a = 0
        i = 0
        while i < len(sorted_diff) - 1:
            if sorted_diff[i+1] == sorted_diff[i] + 1:
                a += 1
                i += 2
            else:
                i += 1
        
        cost = a * 1 + (d // 2 - a) * x
        return cost