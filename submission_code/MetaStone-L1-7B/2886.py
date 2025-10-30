class Solution:
    def finalString(self, s: str) -> str:
        current = []
        for c in s:
            current.append(c)
            if c == 'i':
                current = current[::-1]
        return ''.join(current)