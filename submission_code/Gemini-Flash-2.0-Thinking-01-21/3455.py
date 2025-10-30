class Solution:
    def minimumLength(self, s: str) -> int:
        s_list = list(s)
        while True:
            operation_performed = False
            possible_operation = False
            best_i = -1
            best_left_idx = -1
            best_right_idx = -1
            found_operation = False
            for i in range(len(s_list)):
                char_at_i = s_list[i]
                left_idx = -1
                for j in range(i - 1, -2, -1):
                    if j >= 0 and s_list[j] == char_at_i:
                        left_idx = j
                        break
                right_idx = -1
                for j in range(i + 1, len(s_list)):
                    if s_list[j] == char_at_i:
                        right_idx = j
                        break
                if left_idx != -1 and right_idx != -1:
                    best_i = i
                    best_left_idx = left_idx
                    best_right_idx = right_idx
                    found_operation = True
                    break
            if found_operation:
                del s_list[max(best_left_idx, best_right_idx)]
                del s_list[min(best_left_idx, best_right_idx)]
                operation_performed = True
            else:
                break
            if not operation_performed:
                break
        return len(s_list)