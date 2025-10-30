class Solution:
    def isFascinating(self, n: int) -> bool:
        # Concatenate n, 2*n, and 3*n
        concatenated_str = str(n) + str(2 * n) + str(3 * n)

        # Check if the concatenated string has exactly 9 characters
        if len(concatenated_str) != 9:
            return False

        # Check if the concatenated string contains all digits from 1 to 9 exactly once
        digit_set = set(concatenated_str)
        return digit_set == set('123456789')