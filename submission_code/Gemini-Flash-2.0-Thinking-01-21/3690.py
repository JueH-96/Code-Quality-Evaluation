import math

class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        def is_possible(max_length):
            operations_needed = 0
            i = 0
            while i < n:
                j = i
                while j < n and s[j] == s[i]:
                    j += 1
                run_length = j - i
                if run_length > max_length:
                    operations_needed += (run_length - max_length)
                i = j
            return operations_needed <= numOps

        low = 1
        high = n
        min_len = n
        while low <= high:
            mid_len = (low + high) // 2
            if is_possible_with_ops(mid_len, s, numOps):
                min_len = mid_len
                high = mid_len - 1
            else:
                low = mid_len + 1
        return min_len

    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        def is_possible_with_ops(max_length, original_s, allowed_ops):
            operations_needed = 0
            current_s_list = list(original_s)
            i = 0
            while i < n:
                j = i
                while j < n and current_s_list[j] == current_s_list[i]:
                    j += 1
                run_length = j - i
                if run_length > max_length:
                    ops = run_length - max_length
                    operations_needed += ops
                    if operations_needed > allowed_ops:
                        return False
                    for k in range(i + max_length, j):
                        if (k - (i + max_length)) % 2 == 0:
                            current_s_list[k] = '1' if current_s_list[k] == '0' else '0'
                i = j
            return operations_needed <= allowed_ops

        low = 1
        high = n
        min_len = n
        while low <= high:
            mid_len = (low + high) // 2
            if is_possible_with_ops(mid_len, s, numOps):
                min_len = mid_len
                high = mid_len - 1
            else:
                low = mid_len + 1
        return min_len