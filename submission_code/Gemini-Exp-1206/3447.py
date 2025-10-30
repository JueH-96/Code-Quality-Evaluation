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
                s_list.pop(digit_index)
                s_list.pop(non_digit_index)
            else:
                s_list.pop(digit_index)

        return "".join(s_list)