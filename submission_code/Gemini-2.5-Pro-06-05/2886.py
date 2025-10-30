class Solution:
    def finalString(self, s: str) -> str:
        """
        Calculates the final string on the screen after typing the input string 's'
        on a faulty keyboard where 'i' reverses the current text.
        """
        
        # Use a list of characters to build the final string.
        # This is more efficient for modifications like append and reverse
        # compared to using Python's immutable strings directly in a loop.
        result_chars = []
        
        # Iterate over each character in the input string.
        for char in s:
            if char == 'i':
                # If the character is 'i', reverse the list of characters.
                # The reverse() method modifies the list in-place.
                result_chars.reverse()
            else:
                # Otherwise, append the character to the list.
                result_chars.append(char)
        
        # Join the characters in the list to form the final string.
        return "".join(result_chars)