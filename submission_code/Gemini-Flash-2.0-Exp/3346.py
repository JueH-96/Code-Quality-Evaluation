class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        n = len(s)
        t = list(s)
        for i in range(n):
            dist = min(abs(ord(s[i]) - ord('a')), 26 - abs(ord(s[i]) - ord('a')))
            if k >= dist:
                t[i] = 'a'
                k -= dist
            else:
                diff = k
                if ord(s[i]) - diff >= ord('a'):
                    t[i] = chr(ord(s[i]) - diff)
                else:
                    rem = diff - (ord(s[i]) - ord('a'))
                    t[i] = chr(ord('z') - rem + 1)
                k = 0
                break
        return "".join(t)