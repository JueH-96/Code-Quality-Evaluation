class Solution:
    def minimumSteps(self, s: str) -> int:
        n = len(s)
        ones = [i for i, c in enumerate(s) if c == '1']
        m = len(ones)
        if m == 0:
            return 0
        total = 0
        for i in range(m):
            target = (n - m) + i
            total += (target - ones[i])
        return total