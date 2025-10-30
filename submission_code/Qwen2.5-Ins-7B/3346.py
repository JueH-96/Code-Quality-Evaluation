class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        s = list(s)
        for i in range(len(s)):
            min_change = min(26 - (ord(s[i]) - ord('a')), ord('a') + k)
            if min_change == 26:
                s[i] = 'a'
                k -= 25
            else:
                s[i] = chr(ord(s[i]) + min_change)
                k -= min_change
            if k < 0:
                break
        return ''.join(s)