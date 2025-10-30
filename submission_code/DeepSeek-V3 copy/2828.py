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
        # Perform the operation from i to the end or until another 'a' is found
        j = i
        while j < n and s[j] != 'a':
            s[j] = chr(ord(s[j]) - 1)
            j += 1
        return ''.join(s)