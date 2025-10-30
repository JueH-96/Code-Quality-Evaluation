class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        last_non_empty = ""
        while s:
            last_non_empty = s
            for char in 'abcdefghijklmnopqrstuvwxyz':
                if char in s:
                    s = s.replace(char, '', 1)
        return last_non_empty