class Solution:
    def smallestString(self, s: str) -> str:
        # Convert s into a list of characters for in-place modifications
        s_list = list(s)
        n = len(s_list)
        # Find the leftmost character that is not 'a'
        i = 0
        while i < n and s_list[i] == 'a':
            i += 1
        # If every character is 'a', modify the last one to 'z'
        if i == n:
            s_list[-1] = 'z'
            return "".join(s_list)
        
        # Else, from the first non-'a' character, keep decrementing characters
        while i < n and s_list[i] != 'a':
            # Decrement the character (if it is 'a', then it would wrap to 'z', but we are not including that according to our loop condition)
            s_list[i] = chr(ord(s_list[i]) - 1)
            i += 1
            
        return "".join(s_list)