class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        # Pointer for str2
        j = 0
        n2 = len(str2)
        
        # Function to get the next cyclic character
        def next_char(c):
            return 'a' if c == 'z' else chr(ord(c) + 1)
        
        # Go through each character in str1
        for c in str1:
            if j < n2:
                # If the current char matches str2[j], consume it
                if c == str2[j]:
                    j += 1
                # Or if incrementing c matches str2[j], also consume it
                elif next_char(c) == str2[j]:
                    j += 1
                    
            # If we've matched all of str2, no need to check further
            if j == n2:
                return True
        
        return j == n2