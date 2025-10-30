class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        sum_digits = sum(int(d) for d in str(x))
        return sum_digits if x % sum_digits == 0 else -1