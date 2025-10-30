class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        if x == 0:
            return -1
        sum_digits = sum(int(digit) for digit in str(x))
        if sum_digits == 0:
            return -1
        if x % sum_digits == 0:
            return sum_digits
        else:
            return -1