class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        cost0 = 0
        cost1 = 0
        s0 = list(s)
        s1 = list(s)

        for i in range(n):
            if s0[i] == '1':
                s0[i] = '0'
                cost0 += min(i + 1, n - i)
            if s1[i] == '0':
                s1[i] = '1'
                cost1 += min(i + 1, n - i)
        
        return min(cost0, cost1)