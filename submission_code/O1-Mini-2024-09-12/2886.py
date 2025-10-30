class Solution:
    def finalString(self, s: str) -> str:
        t = []
        for c in s:
            if c == 'i':
                t.reverse()
            else:
                t.append(c)
        return ''.join(t)