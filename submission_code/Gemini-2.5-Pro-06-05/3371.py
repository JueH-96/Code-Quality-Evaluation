class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        """
        Calculates the sum of the digits of x. If x is a Harshad number 
        (divisible by the sum of its digits), it returns the sum.
        Otherwise, it returns -1.
        
        Args:
            x: An integer between 1 and 100.
            
        Returns:
            The sum of the digits of x if x is a Harshad number, otherwise -1.
        """
        
        # Step 1: Calculate the sum of the digits of x.
        # We can convert the number to a string to easily access its digits.
        # A generator expression with sum() is a concise way to do this.
        digit_sum = sum(int(digit) for digit in str(x))
        
        # Step 2: Check if x is divisible by the sum of its digits.
        # A number is divisible by another if the remainder of their division is 0.
        if x % digit_sum == 0:
            # If it is a Harshad number, return the sum of the digits.
            return digit_sum
        else:
            # Otherwise, return -1.
            return -1

        # This logic can also be written more concisely using a ternary operator:
        # return digit_sum if x % digit_sum == 0 else -1