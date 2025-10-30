class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        
        # locate the first character that is not 'a'
        i = 0
        while i < n and s[i] == 'a':
            i += 1
        
        # the whole string is 'a' → we have to change exactly one char;
        # changing the last one (furthest to the right) to 'z' is optimal
        if i == n:
            return s[:-1] + 'z'
        
        # otherwise: starting from the first non-'a', keep moving left
        # by one position in the alphabet until an 'a' appears or we run
        # out of characters
        res = list(s)
        while i < n and res[i] != 'a':
            res[i] = chr(ord(res[i]) - 1)  # previous character; never 'a' here
            i += 1
        
        return ''.join(res)