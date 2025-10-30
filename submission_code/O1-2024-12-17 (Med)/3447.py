class Solution:
    def clearDigits(self, s: str) -> str:
        # Convert the string into a list of characters for easier manipulation
        chars = list(s)
        
        while True:
            # Find the index of the first digit
            digit_index = None
            for i, c in enumerate(chars):
                if c.isdigit():
                    digit_index = i
                    break
            
            # If no digit is found, we're done
            if digit_index is None:
                break
            
            # Find the closest non-digit character to the left of the digit
            for j in range(digit_index - 1, -1, -1):
                if not chars[j].isdigit():
                    # Remove the digit first, then the closest non-digit
                    chars.pop(digit_index)  # pops the first digit found
                    chars.pop(j)            # pops the closest non-digit before it
                    break
        
        return ''.join(chars)