class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        sum_digits = sum(int(d) for d in str(x))
        if x % sum_digits == 0:
            return sum_digits
        else:
            return -1