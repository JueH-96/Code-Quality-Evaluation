class Solution:
    def clearDigits(self, s: str) -> str:
        s_list = list(s)
        while True:
            # Find the first occurrence of a digit
            first_digit_index = None
            for i in range(len(s_list)):
                if s_list[i].isdigit():
                    first_digit_index = i
                    break
            if first_digit_index is None:
                break  # No more digits to process

            # Find the closest non-digit to the left of the first digit
            first_non_digit_index = None
            for j in range(first_digit_index - 1, -1, -1):
                if not s_list[j].isdigit():
                    first_non_digit_index = j
                    break

            # Remove both characters
            del s_list[first_digit_index]
            del s_list[first_non_digit_index]

        return ''.join(s_list)