class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        """
        Returns the sum of digits of x if x is a Harshad number (i.e., x is divisible by
        the sum of its digits), otherwise returns -1.
        """
        # Compute the sum of the digits
        digit_sum = sum(int(d) for d in str(x))
        
        # Check Harshad property and return accordingly
        if x % digit_sum == 0:
            return digit_sum
        else:
            return -1