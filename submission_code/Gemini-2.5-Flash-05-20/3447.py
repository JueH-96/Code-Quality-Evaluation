class Solution:
    def clearDigits(self, s: str) -> str:
        # Use a list to simulate a stack.
        # This stack will store the non-digit characters that are currently "kept".
        result_chars = [] 

        for char in s:
            if char.isdigit():
                # If the current character is a digit, it is to be removed.
                # Additionally, the "closest non-digit character to its left" must also be removed.
                # In our stack model, this corresponds to the last non-digit character added.
                # The problem guarantees that it's always possible to delete all digits,
                # which implies result_chars will not be empty when a digit is encountered.
                result_chars.pop()
            else:
                # If the current character is a non-digit, add it to our stack.
                # It might be removed later if a digit appears after it.
                result_chars.append(char)
        
        # After processing all characters, join the remaining characters in the stack
        # to form the final string.
        return "".join(result_chars)