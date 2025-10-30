class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        
        def get_digit_product(num: int) -> int:
            """
            Helper function to calculate the product of the digits of a number.
            """
            # Convert the number to a string to easily access its digits.
            s_num = str(num)
            
            # If '0' is a digit, the product will be 0.
            if '0' in s_num:
                return 0
                
            product = 1
            for digit_char in s_num:
                product *= int(digit_char)
            return product

        # Start checking numbers from n upwards.
        current_num = n
        while True:
            # Calculate the product of digits for the current number.
            product = get_digit_product(current_num)
            
            # Check if the product is divisible by t.
            # A number 'a' is divisible by 'b' if a % b == 0.
            if product % t == 0:
                # We have found the smallest number >= n satisfying the condition.
                return current_num
            
            # If the condition is not met, move to the next integer.
            current_num += 1