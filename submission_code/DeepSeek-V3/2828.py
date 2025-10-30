class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        s = list(s)
        i = 0
        # Find the first non 'a' character
        while i < n and s[i] == 'a':
            i += 1
        if i == n:
            # All characters are 'a', change the last one to 'z'
            s[-1] = 'z'
            return ''.join(s)
        # Now, find the end of the substring to change
        j = i
        while j < n and s[j] != 'a':
            j += 1
        # Change the characters from i to j-1
        for k in range(i, j):
            if s[k] == 'a':
                s[k] = 'z'
            else:
                s[k] = chr(ord(s[k]) - 1)
        return ''.join(s)