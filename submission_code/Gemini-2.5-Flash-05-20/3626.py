class Solution:
    def _product_of_digits(self, num: int) -> int:
        """
        Helper method to calculate the product of digits of a given number.
        Returns 0 if any digit is 0, as the product would be 0.
        """
        s = str(num)
        product = 1
        for char_digit in s:
            digit = int(char_digit)
            if digit == 0:
                # If any digit is 0, the entire product is 0.
                # 0 is divisible by any positive integer t.
                return 0
            product *= digit
        return product

    def smallestNumber(self, n: int, t: int) -> int:
        """
        Returns the smallest number greater than or equal to n
        such that the product of its digits is divisible by t.
        """
        current_num = n
        
        # Iterate starting from n, incrementing by 1
        # until the condition is met.
        # Given the constraints (n <= 100, t <= 10), this loop will terminate quickly.
        while True:
            prod = self._product_of_digits(current_num)
            
            # Check if the product of digits is divisible by t
            if prod % t == 0:
                return current_num
            
            # If not divisible, check the next number
            current_num += 1