class Solution:
    def maxOperations(self, s: str) -> int:
        n = len(s)
        ones = 0
        ans = 0
        for i in range(n):
            if s[i] == '1':
                ones += 1
            elif ones > 0:
                ans += ones
        return ans