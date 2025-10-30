class Solution:
    def minChanges(self, s: str) -> int:
        ans = 0
        prev = s[0]
        for i in range(1, len(s)):
            if s[i] != prev:
                ans += 1
                prev = s[i]
        return ans // 2