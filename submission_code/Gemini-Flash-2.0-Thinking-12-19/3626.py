class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def digit_product(num):
            product = 1
            for digit in str(num):
                product *= int(digit)
            return product

        current_num = n
        while True:
            prod = digit_product(current_num)
            if prod % t == 0:
                return current_num
            current_num += 1