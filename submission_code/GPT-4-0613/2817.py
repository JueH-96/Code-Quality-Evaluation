class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        prefix = [0] * (n + 1)
        suffix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + (s[i - 1] == '1')
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + (s[i] == '0')
        return min(prefix[i] + suffix[i + 1] for i in range(n)) + min(prefix[i] + suffix[i] for i in range(1, n + 1))