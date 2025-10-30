class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        # Convert the string to a list of characters for easy manipulation
        s_list = list(s)
        
        # Find the first character that is not 'a' to start the operation
        i = 0
        while i < n and s_list[i] == 'a':
            i += 1
        
        # If all characters are 'a', change the last one to 'z'
        if i == n:
            s_list[-1] = 'z'
        else:
            # Start changing characters from the first non-'a' character
            while i < n and s_list[i] != 'a':
                s_list[i] = chr(ord(s_list[i]) - 1)
                i += 1
        
        # Convert the list back to a string and return
        return ''.join(s_list)