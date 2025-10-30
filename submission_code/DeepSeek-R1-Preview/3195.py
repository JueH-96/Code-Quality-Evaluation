class Solution:
    def minimumSteps(self, s: str) -> int:
        ones = []
        n = len(s)
        for i in range(n):
            if s[i] == '1':
                ones.append(i)
        k = len(ones)
        if k == 0:
            return 0
        total = 0
        for i in range(k):
            target = (n - k) + i
            total += (target - ones[i])
        return total