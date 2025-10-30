class Solution:
    def clearDigits(self, s: str) -> str:
        s_list = list(s)
        while True:
            digit_index = -1
            for i in range(len(s_list)):
                if s_list[i].isdigit():
                    digit_index = i
                    break
            if digit_index == -1:
                break

            non_digit_index = -1
            for i in range(digit_index - 1, -1, -1):
                if not s_list[i].isdigit():
                    non_digit_index = i
                    break

            if non_digit_index != -1:
                del s_list[non_digit_index]
                del s_list[digit_index - 1] # index shifted after deleting previous element
            else:
                del s_list[digit_index]
        return "".join(s_list)