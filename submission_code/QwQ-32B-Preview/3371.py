class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        original_x = x
        sum_digits = 0
        while x > 0:
            sum_digits += x % 10
            x = x // 10
        if original_x % sum_digits == 0:
            return sum_digits
        return -1