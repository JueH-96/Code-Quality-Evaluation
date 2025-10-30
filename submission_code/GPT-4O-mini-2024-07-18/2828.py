class Solution:
    def smallestString(self, s: str) -> str:
        # Initialize the result as a list of characters from the original string
        result = list(s)
        n = len(s)
        
        # Iterate through the string to find the first character that can be decremented
        for i in range(n):
            if s[i] > 'a':  # If the character is greater than 'a', we can decrement it
                # Decrement the substring from the start to the current index
                for j in range(i + 1):
                    result[j] = chr(ord(s[j]) - 1) if s[j] > 'a' else 'z'
                break
        
        return ''.join(result)