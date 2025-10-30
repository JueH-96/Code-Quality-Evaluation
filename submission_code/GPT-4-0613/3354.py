class Solution:
    def minimizeStringValue(self, s: str) -> str:
        s = list(s)
        for i in range(len(s)):
            if s[i] == '?':
                for ch in 'abcdefghijklmnopqrstuvwxyz':
                    if (i > 0 and s[i - 1] == ch) or (i < len(s) - 1 and s[i + 1] == ch):
                        continue
                    s[i] = ch
                    break
        return ''.join(s)