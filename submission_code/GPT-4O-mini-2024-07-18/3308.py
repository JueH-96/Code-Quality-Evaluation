class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        while s:
            new_s = s
            for char in 'abcdefghijklmnopqrstuvwxyz':
                new_s = new_s.replace(char, '', 1)
            if new_s == s:
                break
            s = new_s
        return s