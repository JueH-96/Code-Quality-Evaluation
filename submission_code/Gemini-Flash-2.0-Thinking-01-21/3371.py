class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        temp_x = x
        sum_digits = 0
        while temp_x > 0:
            digit = temp_x % 10
            sum_digits += digit
            temp_x //= 10
        if x % sum_digits == 0:
            return sum_digits
        else:
            return -1