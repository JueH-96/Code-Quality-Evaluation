class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        while True:
            new_s = list(s)
            for char in 'abcdefghijklmnopqrstuvwxyz':
                if char in new_s:
                    new_s.remove(char)
            new_s = ''.join(new_s)
            if not new_s:
                return s
            s = new_s