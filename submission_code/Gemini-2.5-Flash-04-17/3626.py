class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        
        def get_digit_product(num):
            """Calculates the product of digits of a number."""
            # If the number contains a 0 digit, the product is 0.
            # This is true even if the input num itself is 0, but n >= 1.
            # For num > 0, if any digit is 0, the product becomes 0.
            
            s_num = str(num)
            product = 1
            for digit_char in s_num:
                digit = int(digit_char)
                if digit == 0:
                    return 0 # Product is 0 if any digit is 0
                product *= digit
            return product

        current_number = n
        while True:
            product = get_digit_product(current_number)
            
            # Check if the product of digits is divisible by t
            if product % t == 0:
                return current_number
            
            # If not, check the next number
            current_number += 1