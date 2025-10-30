class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Initialize a variable to store the result before the last operation
        result_before_last = ""
        
        # Loop until the string becomes empty
        while s:
            # Store the current state of the string before the operation
            result_before_last = s
            # Iterate over each character from 'a' to 'z'
            for c in range(ord('a'), ord('z') + 1):
                # Find the first occurrence of the character
                i = s.find(chr(c))
                # If the character is found, remove it from the string
                if i != -1:
                    s = s[:i] + s[i+1:]
            # If no characters were removed, break the loop
            if result_before_last == s:
                break
        
        # Return the result before the last operation
        return result_before_last