class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def get_digit_product(num):
            # Calculate product of digits
            product = 1
            for digit in str(num):
                product *= int(digit)
            return product
        
        # Start checking from n
        current = n
        
        # Keep checking numbers until we find one that satisfies our condition
        while True:
            # Get product of digits for current number
            product = get_digit_product(current)
            
            # If product is 0, it's only divisible by 1
            if product == 0:
                if t == 1:
                    return current
            # If product is not 0, check if it's divisible by t
            elif product % t == 0:
                return current
                
            current += 1