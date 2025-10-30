class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        res = float('inf')
        # Calculate the cost to make all characters equal from index 0 to i
        for i in range(2):
            cost = 0
            for j in range(n):
                if j > i and s[j] == '1':
                    cost += j + 1
                elif j < i and s[j] == '0':
                    cost += n - j
            res = min(res, cost)
        # Calculate the cost to make all characters equal from index i to n
        for i in range(n):
            cost = 0
            for j in range(n):
                if j < i and s[j] == '0':
                    cost += i
                elif j > i and s[j] == '1':
                    cost += n - j + i
            res = min(res, cost)
        return res