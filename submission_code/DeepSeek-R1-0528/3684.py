class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        parts = p.split('*', 1)
        pre = parts[0]
        suf = parts[1]
        L = len(pre)
        M = len(suf)
        if L == 0 and M == 0:
            return True
        n = len(s)
        for i in range(n - L + 1):
            if s[i:i+L] != pre:
                continue
            if M == 0:
                return True
            idx = s.find(suf, i + L)
            if idx != -1:
                return True
        return False