class Solution:
    def finalString(self, s: str) -> str:
        # Initialize an empty string to store the result
        result = ''

        # Iterate through the given string
        for i, char in enumerate(s):
            # If the character is 'i', reverse the result so far
            if char == 'i':
                result = result[::-1]
            else:
                # Otherwise, add the character to the result
                result += char
        
        # Return the final result after iterating through the string
        return result