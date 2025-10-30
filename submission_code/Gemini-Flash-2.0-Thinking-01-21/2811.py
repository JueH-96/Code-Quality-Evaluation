class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        result_array = []
        current_number = 1
        while len(result_array) < n:
            is_safe = True
            for num in result_array:
                if num + current_number == k:
                    is_safe = False
                    break
            if is_safe:
                result_array.append(current_number)
            current_number += 1
        return sum(result_array)