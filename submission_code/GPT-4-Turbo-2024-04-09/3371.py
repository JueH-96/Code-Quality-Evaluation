class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        def digit_sum(n):
            return sum(int(d) for d in str(n))
        
        sum_digits = digit_sum(x)
        if x % sum_digits == 0:
            return sum_digits
        else:
            return -1