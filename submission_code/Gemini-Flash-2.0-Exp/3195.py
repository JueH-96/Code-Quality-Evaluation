class Solution:
    def minimumSteps(self, s: str) -> int:
        n = len(s)
        ans = 0
        white_count = 0
        for i in range(n):
            if s[i] == '0':
                ans += (i - white_count)
                white_count += 1
        return ans