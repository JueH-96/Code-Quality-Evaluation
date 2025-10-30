class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        cost1 = 0
        cost2 = 0
        curr = s[0]
        for i in range(n):
            if s[i] != curr:
                if curr == '0':
                    cost1 += i + 1
                    curr = '1'
                else:
                    cost2 += n - i
                    curr = '0'

        
        curr = s[0]
        cost3 = 0
        cost4 = 0
        for i in range(n):
            if s[i] != curr:
                if curr == '1':
                    cost3 += i + 1
                    curr = '0'
                else:
                    cost4 += n - i
                    curr = '1'

        return min(cost1,cost2,cost3,cost4)