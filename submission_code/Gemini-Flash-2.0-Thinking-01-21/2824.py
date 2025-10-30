class Solution:
    def isFascinating(self, n: int) -> bool:
        n2 = 2 * n
        n3 = 3 * n
        concatenated_string = str(n) + str(n2) + str(n3)
        if '0' in concatenated_string:
            return False
        digit_set = set(concatenated_string)
        if len(digit_set) != 9:
            return False
        sorted_digits = sorted(list(digit_set))
        expected_digits = [str(i) for i in range(1, 10)]
        if sorted_digits == expected_digits:
            return True
        else:
            return False