class Solution:
    def minimumSteps(self, s: str) -> int:
        n = len(s)
        ans = 0
        cnt = 0
        for i in range(n):
            if s[i] == '1':
                cnt += 1
            else:
                ans += cnt
        return ans