class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # Initialize a list to store the result
        result = []

        # Iterate through the string
        for char in s:
            if char == '?':
                # If the character is '?', append the next available smallest letter
                if not result or result[-1] == 'z':
                    result.append('a')
                else:
                    result.append(chr(ord(result[-1]) + 1))
            else:
                # If the character is not '?', append it to the result
                result.append(char)

        # Join the list into a string and return
        return ''.join(result)