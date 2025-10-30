class Solution:
    def clearDigits(self, s: str) -> str:
        char_list = list(s)
        while True:
            digit_index = -1
            for i in range(len(char_list)):
                if char_list[i].isdigit():
                    digit_index = i
                    break
            if digit_index == -1:
                break
            non_digit_index = -1
            for i in range(digit_index - 1, -1, -1):
                if not char_list[i].isdigit():
                    non_digit_index = i
                    break
            if non_digit_index != -1:
                if non_digit_index < digit_index:
                    char_list.pop(non_digit_index)
                    char_list.pop(digit_index - 1)
            else:
                char_list.pop(digit_index)
        return "".join(char_list)