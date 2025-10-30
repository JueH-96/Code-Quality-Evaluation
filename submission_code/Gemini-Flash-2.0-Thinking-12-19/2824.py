class Solution:
    def isFascinating(self, n: int) -> bool:
        num2 = 2 * n
        num3 = 3 * n
        concatenated_str = str(n) + str(num2) + str(num3)
        digit_counts = {}
        for digit in concatenated_str:
            digit_counts[digit] = digit_counts.get(digit, 0) + 1

        if '0' in digit_counts:
            return False

        if len(digit_counts) != 9:
            return False

        for digit in '123456789':
            if digit not in digit_counts or digit_counts[digit] != 1:
                return False
        return True