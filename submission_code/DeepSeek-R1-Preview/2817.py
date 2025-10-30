class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        transitions = []
        for i in range(n - 1):
            if s[i] != s[i + 1]:
                transitions.append(i)
        total = 0
        for i in transitions:
            cost = min(i + 1, n - i)
            total += cost
        return total