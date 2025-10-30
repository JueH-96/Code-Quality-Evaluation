class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        def distance(c1, c2):
            return min(abs(ord(c1) - ord(c2)), 26 - abs(ord(c1) - ord(c2)))

        n = len(s)
        t = list(s)

        for i in range(n):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if distance(s[i], c) <= k:
                    t[i] = c
                    k -= distance(s[i], c)
                    break

        return ''.join(t)