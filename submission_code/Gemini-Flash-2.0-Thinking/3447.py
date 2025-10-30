class Solution:
    def clearDigits(self, s: str) -> str:
        s_list = list(s)
        while any(char.isdigit() for char in s_list):
            first_digit_index = -1
            for i, char in enumerate(s_list):
                if char.isdigit():
                    first_digit_index = i
                    break

            if first_digit_index != -1:
                non_digit_index = -1
                for i in range(first_digit_index - 1, -1, -1):
                    if not s_list[i].isdigit():
                        non_digit_index = i
                        break

                if non_digit_index != -1:
                    # Remove the elements at higher index first to avoid index issues
                    indices_to_remove = sorted([first_digit_index, non_digit_index], reverse=True)
                    for index in indices_to_remove:
                        if 0 <= index < len(s_list):
                            del s_list[index]
        return "".join(s_list)