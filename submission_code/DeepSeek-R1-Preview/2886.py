class Solution:
    def finalString(self, s: str) -> str:
        current = []
        for c in s:
            if c == 'i':
                current.reverse()
            else:
                current.append(c)
        return ''.join(current)