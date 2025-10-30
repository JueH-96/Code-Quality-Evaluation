class Solution:
    def smallestString(self, s: str) -> str:
        # Convert to list for easy in-place edits
        chars = list(s)
        n = len(chars)
        
        # Find the first character that is not 'a'
        i = 0
        while i < n and chars[i] == 'a':
            i += 1
        
        # If the entire string is 'a's, we must decrement exactly one substring:
        # best to pick the last character, turning it from 'a' to 'z'
        if i == n:
            chars[-1] = 'z'
            return ''.join(chars)
        
        # Otherwise, from the first non-'a' char, keep decrementing
        # until we hit an 'a' or end of string
        while i < n and chars[i] != 'a':
            # decrement character by one in the alphabet
            chars[i] = chr(ord(chars[i]) - 1)
            i += 1
        
        return ''.join(chars)