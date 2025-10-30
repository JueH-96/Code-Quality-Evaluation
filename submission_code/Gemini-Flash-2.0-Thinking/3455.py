class Solution:
    def minimumLength(self, s: str) -> int:
        s_list = list(s)

        def find_closest(arr, val):
            if not arr:
                return -1
            return min(arr, key=lambda x: abs(x - val))

        while True:
            found_operation = False
            best_operation = None

            for i in range(len(s_list)):
                char = s_list[i]

                left_indices = [j for j in range(i) if s_list[j] == char]
                right_indices = [k for k in range(i + 1, len(s_list)) if s_list[k] == char]

                if left_indices and right_indices:
                    left_closest = min(left_indices, key=lambda x: i - x)
                    right_closest = min(right_indices, key=lambda x: x - i)
                    best_operation = (i, left_closest, right_closest)
                    break

            if not best_operation:
                break

            i, left_remove_idx, right_remove_idx = best_operation

            if left_remove_idx < right_remove_idx:
                s_list.pop(right_remove_idx)
                s_list.pop(left_remove_idx)
            else:
                s_list.pop(left_remove_idx)
                s_list.pop(right_remove_idx)

            found_operation = True

            if not found_operation:
                break

        return len(s_list)