class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        # Helper function to calculate the product of digits of a number
        def digit_product(num):
            product = 1
            for digit in str(num):
                product *= int(digit)
            return product
        
        # Main logic: loop starting from n upwards
        while True:
            if digit_product(n) % t == 0:
                return n
            n += 1