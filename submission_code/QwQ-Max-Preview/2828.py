class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        # Check if all characters are 'a'
        if all(c == 'a' for c in s):
            return s[:-1] + 'z' if n else 'a'
        
        # Find the first non 'a' character
        first = 0
        while first < n and s[first] == 'a':
            first += 1
        
        # Find the end of the consecutive non 'a' substring
        last = first
        while last < n and s[last] != 'a':
            last += 1
        last -= 1  # Adjust to the last index of non 'a'
        
        # Convert the substring to the previous characters
        res = list(s)
        for i in range(first, last + 1):
            res[i] = chr(ord(res[i]) - 1)
        
        return ''.join(res)