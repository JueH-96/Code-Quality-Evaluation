class Solution:
    def minimizeStringValue(self, s: str) -> str:
        n = len(s)
        result = list(s)
        
        # To keep track of the last used character to avoid repetition
        last_used = None
        
        for i in range(n):
            if result[i] == '?':
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if (i == 0 or result[i-1] != c) and (i == n-1 or result[i+1] != c):
                        result[i] = c
                        break
            last_used = result[i]
        
        return ''.join(result)