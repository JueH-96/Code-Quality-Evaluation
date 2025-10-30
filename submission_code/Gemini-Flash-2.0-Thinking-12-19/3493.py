class Solution:
    def maxOperations(self, s: str) -> int:
        s_list = list(s)
        count = 0
        while True:
            operation_made_in_this_pass = False
            i = 0
            while i < len(s_list) - 1:
                if s_list[i] == '1' and s_list[i+1] == '0':
                    char_to_move = '1'
                    current_index = i
                    while current_index + 1 < len(s_list) and s_list[current_index + 1] == '0':
                        s_list[current_index] = '0'
                        s_list[current_index + 1] = char_to_move
                        current_index += 1
                    s_list[i] = '0'
                    # s_list[current_index] = char_to_move  // Already done in the loop
                    count += 1
                    operation_made_in_this_pass = True
                    break # Restart from the beginning
                i += 1
            if not operation_made_in_this_pass:
                break
        return count