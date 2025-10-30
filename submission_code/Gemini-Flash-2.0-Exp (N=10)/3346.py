class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        n = len(s)
        t = list(s)
        for i in range(n):
            diff = abs(ord(s[i]) - ord('a'))
            if k >= diff:
                k -= diff
                t[i] = 'a'
            else:
                target_char = chr(ord(s[i]) - k)
                t[i] = target_char
                k = 0
                break
        return "".join(t)