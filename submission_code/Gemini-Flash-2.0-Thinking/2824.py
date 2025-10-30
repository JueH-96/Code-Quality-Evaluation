class Solution:
    def isFascinating(self, n: int) -> bool:
        double_n = 2 * n
        triple_n = 3 * n

        concatenated_string = str(n) + str(double_n) + str(triple_n)

        if len(concatenated_string) != 9:
            return False

        if '0' in concatenated_string:
            return False

        seen_digits = set()
        for digit in concatenated_string:
            if digit in seen_digits:
                return False  # Duplicate digit
            seen_digits.add(digit)

        required_digits = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}
        return seen_digits == required_digits