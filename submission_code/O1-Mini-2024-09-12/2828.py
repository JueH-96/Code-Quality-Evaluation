class Solution:
    def smallestString(self, s: str) -> str:
        s = list(s)
        n = len(s)
        started = False
        i = 0
        while i < n:
            if not started and s[i] > 'a':
                started = True
                s[i] = chr(ord(s[i]) - 1)
            elif started and s[i] > 'a':
                s[i] = chr(ord(s[i]) - 1)
            else:
                break
            i += 1
        if not started:
            s[-1] = 'z'
        return ''.join(s)