class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def product_of_digits(num):
            product = 1
            for digit in str(num):
                product *= int(digit)
            return product

        current_num = n
        while True:
            if product_of_digits(current_num) % t == 0:
                return current_num
            current_num += 1