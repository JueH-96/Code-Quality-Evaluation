class Solution:
    def smallestString(self, s: str) -> str:
        if all(c == 'a' for c in s):
            return s[:-1] + 'z'
        
        result = []
        changed = False
        for i, c in enumerate(s):
            if c != 'a' and not changed:
                while i < len(s) and s[i] != 'a':
                    result.append(chr(ord(s[i]) - 1))
                    i += 1
                changed = True
                result.append(s[i:])
                break
            result.append(c)
        
        return ''.join(result)