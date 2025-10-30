class Solution:
    def smallestString(self, s: str) -> str:
        s = list(s)
        n = len(s)
        i = 0
        while i < n and s[i] == 'a':
            i += 1
        
        if i == n:
            s[-1] = 'z'
        else:
            while i < n:
                while i < n and s[i] != 'a':
                    s[i] = chr(ord(s[i]) - 1)
                    i += 1
                if i < n:
                    s[i] = 'z'
                    while i + 1 < n and s[i+1] == 'a':
                        s[i+1] = 'z'
                        i += 1
                break
        return "".join(s)