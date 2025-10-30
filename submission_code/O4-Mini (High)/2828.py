class Solution:
    def smallestString(self, s: str) -> str:
        # Convert to list for in-place edits
        s_lst = list(s)
        n = len(s_lst)
        
        # Skip all leading 'a's
        i = 0
        while i < n and s_lst[i] == 'a':
            i += 1
        
        # If the whole string was 'a's, change the last one to 'z'
        if i == n:
            s_lst[n - 1] = 'z'
            return ''.join(s_lst)
        
        # From the first non-'a', decrement each character until we hit an 'a' or end
        j = i
        while j < n and s_lst[j] != 'a':
            # shift one character back in the alphabet
            s_lst[j] = chr(ord(s_lst[j]) - 1)
            j += 1
        
        return ''.join(s_lst)