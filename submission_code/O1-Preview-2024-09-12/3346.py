class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        t = list(s)
        n = len(s)
        for i in range(n):
            for c_ord in range(ord('a'), ord('z')+1):
                c = chr(c_ord)
                diff = abs(ord(s[i]) - ord(c))
                cost = min(diff, 26 - diff)
                if cost <= k:
                    t[i] = c
                    k -= cost
                    break  # Found the smallest possible c for this position
        return ''.join(t)