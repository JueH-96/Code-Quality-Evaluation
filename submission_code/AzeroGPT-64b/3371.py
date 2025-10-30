class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        if 1 <= x <= 100:
            digit_sum = sum(map(int, str(x)))
            return digit_sum if x % digit_sum == 0 else -1
        return -1