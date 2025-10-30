class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        sum_digits = 0
        temp = x
        while temp > 0:
            sum_digits += temp % 10
            temp //= 10
        
        if x % sum_digits == 0:
            return sum_digits
        else:
            return -1