class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        s = 0
        temp = x
        while temp > 0:
            digit = temp % 10
            s += digit
            temp //= 10

        if s == 0:
            return -1  # Avoid division by zero

        if x % s == 0:
            return s
        else:
            return -1