class Solution:
    def clearDigits(self, s: str) -> str:
        current_str = list(s)
        while True:
            # Find the first digit in the current string
            digit_idx = -1
            for i in range(len(current_str)):
                if current_str[i].isdigit():
                    digit_idx = i
                    break
            if digit_idx == -1:
                break  # No more digits, exit loop
            
            # Find the closest non-digit to the left
            left_idx = -1
            for i in range(digit_idx - 1, -1, -1):
                if not current_str[i].isdigit():
                    left_idx = i
                    break
            
            # Delete the digit and then the left non-digit character
            # Delete in reverse order of their positions to maintain correct indices
            del current_str[digit_idx]
            del current_str[left_idx]
        
        return ''.join(current_str)