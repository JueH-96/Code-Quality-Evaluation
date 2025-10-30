class Solution:
    def clearDigits(self, s: str) -> str:
        s_list = list(s)
        while True:
            # Find the first occurrence of a digit
            first_digit_idx = None
            for idx in range(len(s_list)):
                if s_list[idx].isdigit():
                    first_digit_idx = idx
                    break
            if first_digit_idx is None:
                break  # No more digits, exit loop
            
            # Find the closest non-digit to the left of the first digit
            non_digit_idx = None
            for j in range(first_digit_idx - 1, -1, -1):
                if not s_list[j].isdigit():
                    non_digit_idx = j
                    break
            
            # Remove the non-digit and the digit
            if non_digit_idx is not None:
                del s_list[non_digit_idx]
                del s_list[first_digit_idx - 1]  # Adjust index after first deletion
        
        return ''.join(s_list)