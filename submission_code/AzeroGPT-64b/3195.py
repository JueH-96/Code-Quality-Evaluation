class Solution:
    def minimumSteps(self, s: str) -> int:
        ans = 0
        lastOne = s.find('1')
        if lastOne == -1:
            return 0
        for idx, ch in enumerate(s):
            if ch == '0' and s[idx:].find('1') != -1:
                ans += 1
            elif ch == '1':
                lastOne = idx
        return ans