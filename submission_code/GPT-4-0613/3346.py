class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        s = list(s)
        for i in range(len(s)):
            if k > 0:
                distance = min(ord(s[i]) - ord('a'), 26 - (ord(s[i]) - ord('a')))
                if k >= distance:
                    s[i] = 'a'
                    k -= distance
                else:
                    s[i] = chr(ord(s[i]) - k)
                    k = 0
        return ''.join(s)