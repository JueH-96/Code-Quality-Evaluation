class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # Convert the string to a list for easy manipulation
        s = list(s)
        n = len(s)
        
        # Iterate over the string
        for i in range(n):
            if s[i] == '?':
                # Find the smallest character that can be placed at position i
                # without increasing the cost
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if (i == 0 or s[i-1] != c) and (i == n-1 or s[i+1] != c):
                        s[i] = c
                        break
        
        # Join the list back into a string
        return ''.join(s)