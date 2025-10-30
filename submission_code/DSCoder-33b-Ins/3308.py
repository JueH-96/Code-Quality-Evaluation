class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        while s:
            for char in alphabet:
                if char in s:
                    s = s.replace(char, '', 1)
            if not s:
                break
        return s