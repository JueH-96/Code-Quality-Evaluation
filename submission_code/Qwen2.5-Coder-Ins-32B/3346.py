class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        def distance(c1, c2):
            d1 = abs(ord(c1) - ord(c2))
            d2 = 26 - d1
            return min(d1, d2)
        
        t = list(s)
        for i in range(len(s)):
            if k == 0:
                break
            for c in 'abcdefghijklmnopqrstuvwxyz':
                d = distance(s[i], c)
                if d <= k:
                    t[i] = c
                    k -= d
                    break
        return ''.join(t)