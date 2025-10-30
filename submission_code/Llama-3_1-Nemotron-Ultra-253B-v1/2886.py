class Solution:
    def finalString(self, s: str) -> str:
        current = ''
        for c in s:
            if c == 'i':
                current = current[::-1]
            else:
                current += c
        return current