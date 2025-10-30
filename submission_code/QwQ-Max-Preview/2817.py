class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        transitions = []
        prev = s[0]
        for i in range(1, n):
            if s[i] != prev:
                transitions.append(i)
                prev = s[i]
        total = 0
        for i in transitions:
            total += min(i, n - i)
        return total