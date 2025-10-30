class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        for c in 'abcdefghijklmnopqrstuvwxyz':
            s = s.replace(c, '', 1)
        return s