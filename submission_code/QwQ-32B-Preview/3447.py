class Solution:
    def clearDigits(self, s: str) -> str:
        s_list = list(s)
        while any(char.isdigit() for char in s_list):
            # Find the index of the first digit
            for i in range(len(s_list)):
                if s_list[i].isdigit():
                    digit_index = i
                    break
            # Find the closest non-digit to the left
            if digit_index > 0 and not s_list[digit_index - 1].isdigit():
                # Delete both the digit and the non-digit to its left
                del s_list[digit_index - 1:digit_index + 1]
            else:
                # Delete only the digit
                del s_list[digit_index]
        return ''.join(s_list)