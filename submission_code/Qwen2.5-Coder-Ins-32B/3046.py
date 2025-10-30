class Solution:
    def minimumOperations(self, num: str) -> int:
        def find_last_digit(target, start_index):
            for i in range(start_index, -1, -1):
                if num[i] == target:
                    return i
            return -1

        n = len(num)
        min_operations = n

        # Check for suffixes '00', '25', '50', '75'
        for suffix in ['00', '25', '50', '75']:
            first = find_last_digit(suffix[1], n - 1)
            if first != -1:
                second = find_last_digit(suffix[0], first - 1)
                if second != -1:
                    min_operations = min(min_operations, n - second - 2)

        # If there's at least one '0', we can make the number '0'
        if '0' in num:
            min_operations = min(min_operations, n - 1)

        return min_operations