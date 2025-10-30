class Solution:
    def minimumOperations(self, num: str) -> int:
        def find_last_occurrence(char):
            return next((i for i in range(len(num) - 1, -1, -1) if num[i] == char), -1)
        
        def count_operations(last_digit, second_last_digit):
            last_idx = find_last_occurrence(last_digit)
            if last_idx == -1:
                return float('inf')
            second_last_idx = find_last_occurrence(second_last_digit)
            if second_last_idx == -1 or second_last_idx > last_idx:
                return float('inf')
            return len(num) - (second_last_idx + 2)
        
        operations = min(
            count_operations('0', '0'),
            count_operations('5', '2'),
            count_operations('5', '7'),
            count_operations('0', '5'),
            len(num) - ('0' in num)  # If no valid pair is found, delete all but one '0' if present
        )
        return operations