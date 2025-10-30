class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        sum_of_digits = sum(int(digit) for digit in str(x))
        if x % sum_of_digits == 0:
            return sum_of_digits
        else:
            return -1