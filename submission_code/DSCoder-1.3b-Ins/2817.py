class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        cost = [0]*n
        min_cost = 0
        for i in range(n):
            if i % 2 == 0:
                if s[i] == '1':
                    min_cost += i + 1
                else:
                    min_cost += n - i
            else:
                if s[i] == '1':
                    min_cost += n - i
                else:
                    min_cost += i + 1
        return min_cost