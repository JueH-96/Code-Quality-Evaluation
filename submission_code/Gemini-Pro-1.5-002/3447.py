class Solution:
    def clearDigits(self, s: str) -> str:
        n = len(s)
        s_list = list(s)
        
        while True:
            digits_indices = [i for i, char in enumerate(s_list) if char.isdigit()]
            if not digits_indices:
                break
            
            digit_index = digits_indices[0]
            
            closest_non_digit_index = -1
            min_dist = float('inf')
            
            for i in range(len(s_list)):
                if not s_list[i].isdigit():
                    dist = abs(i - digit_index)
                    if dist < min_dist:
                        min_dist = dist
                        closest_non_digit_index = i
            
            if closest_non_digit_index != -1:
                if closest_non_digit_index < digit_index:
                    s_list.pop(digit_index)
                    s_list.pop(closest_non_digit_index)
                else:
                    s_list.pop(closest_non_digit_index)
                    s_list.pop(digit_index)
            else:
                s_list.pop(digit_index)


        return "".join(s_list)