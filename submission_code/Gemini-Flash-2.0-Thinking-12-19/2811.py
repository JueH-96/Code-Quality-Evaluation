class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        result_array = []
        selected_numbers = set()
        current_number = 1
        while len(result_array) < n:
            if k - current_number not in selected_numbers:
                result_array.append(current_number)
                selected_numbers.add(current_number)
            current_number += 1
        return sum(result_array)