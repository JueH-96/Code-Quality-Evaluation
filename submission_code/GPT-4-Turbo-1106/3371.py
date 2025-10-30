class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        def sum_of_digits(n):
            return sum(int(digit) for digit in str(n))
        
        digit_sum = sum_of_digits(x)
        return digit_sum if x % digit_sum == 0 else -1