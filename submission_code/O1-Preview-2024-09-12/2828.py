class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        s = list(s)
        i = 0
        while i < n and s[i] == 'a':
            i +=1
        if i == n:
            # All 'a's, decrement the last character
            s[n-1] = 'z'
        else:
            while i < n and s[i] != 'a':
                # Decrease s[i] by one letter
                s[i] = chr(ord(s[i]) - 1) if s[i] != 'a' else 'z'
                i +=1
        return ''.join(s)