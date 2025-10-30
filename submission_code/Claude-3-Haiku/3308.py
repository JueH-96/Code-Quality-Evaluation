class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        for char in 'abcdefghijklmnopqrstuvwxyz':
            s = s.replace(char, '', 1)
        return s