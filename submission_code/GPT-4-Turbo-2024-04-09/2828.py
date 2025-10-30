class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        result = list(s)
        
        # Find the first character that is not 'a'
        for i in range(n):
            if s[i] != 'a':
                # Start changing from this character
                start = i
                while i < n and s[i] != 'a':
                    result[i] = chr(ord(s[i]) - 1) if s[i] != 'a' else 'z'
                    i += 1
                break
        
        return ''.join(result)