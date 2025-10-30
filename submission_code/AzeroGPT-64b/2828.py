class Solution:
    def smallestString(self, s: str) -> str:
        new_s = list(s)
        changed = False
        for i in range(len(s) - 1):
            if s[i] != 'a':
                for j in range(i, len(s)):
                    if s[j] == 'a':
                        break
                    new_s[j] = chr(ord(new_s[j]) - 1)
                changed = True
                break
        
        if not changed:
            new_s[-1] = 'z'
        return ''.join(new_s)