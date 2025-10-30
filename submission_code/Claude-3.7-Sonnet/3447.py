class Solution:
    def clearDigits(self, s: str) -> str:
        while True:
            # Find the first digit
            digit_index = -1
            for i, char in enumerate(s):
                if char.isdigit():
                    digit_index = i
                    break
            
            # If no digit found, break
            if digit_index == -1:
                break
            
            # Find the closest non-digit to its left
            non_digit_index = -1
            for i in range(digit_index - 1, -1, -1):
                if not s[i].isdigit():
                    non_digit_index = i
                    break
            
            # Create a new string without the digit and the non-digit
            s = s[:non_digit_index] + s[non_digit_index+1:digit_index] + s[digit_index+1:]
        
        return s