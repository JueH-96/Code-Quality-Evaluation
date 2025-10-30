class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        sum_digits = sum(int(c) for c in str(x))
        if x % sum_digits == 0:
            return sum_digits
        else:
            return -1