class Solution:
    def clearDigits(self, s: str) -> str:
        """
        Removes all digits and their closest preceding non-digit characters from a string.
        """
        
        # Use a list of characters to build the result. This list will function
        # as a stack (Last-In, First-Out).
        result_chars = []
        
        # Iterate over each character in the input string s.
        for char in s:
            if char.isdigit():
                # The operation is to delete the digit and the closest non-digit
                # to its left. In our left-to-right pass, the closest non-digit
                # that has not been deleted yet is the last one we added to our
                # result list.
                
                # We remove this last-added non-digit character. The problem
                # statement guarantees that for every digit, there's a
                # non-digit to its left to delete, so this pop operation is always valid.
                result_chars.pop()
            else:
                # If the character is a non-digit (a letter), we add it to our list.
                result_chars.append(char)
                
        # After processing all characters, join the remaining characters in the list
        # to form the final string.
        return "".join(result_chars)