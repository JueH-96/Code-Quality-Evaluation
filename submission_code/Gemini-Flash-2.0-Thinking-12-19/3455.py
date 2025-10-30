class Solution:
    def minimumLength(self, s: str) -> int:
        s_list = list(s)
        while True:
            operation_performed = False
            possible_indices = []
            for i in range(len(s_list)):
                char_at_i = s_list[i]
                left_index = -1
                for j in range(i - 1, -2, -1):
                    if j >= 0 and s_list[j] == char_at_i:
                        left_index = j
                        break
                right_index = -1
                for j in range(i + 1, len(s_list)):
                    if s_list[j] == char_at_i:
                        right_index = j
                        break
                if left_index != -1 and right_index != -1:
                    possible_indices.append(i)

            if not possible_indices:
                break

            best_index = -1
            left_to_remove = -1
            right_to_remove = -1

            for i in possible_indices:
                char_at_i = s_list[i]
                left_index = -1
                for j in range(i - 1, -2, -1):
                    if j >= 0 and s_list[j] == char_at_i:
                        left_index = j
                        break
                right_index = -1
                for j in range(i + 1, len(s_list)):
                    if s_list[j] == char_at_i:
                        right_index = j
                        break

                if best_index == -1 or i < best_index:
                    best_index = i
                    left_to_remove = left_index
                    right_to_remove = right_index

            if best_index != -1:
                s_list.pop(right_to_remove)
                s_list.pop(left_to_remove)
                operation_performed = True
            else:
                break
            if not operation_performed:
                break

        return len(s_list)