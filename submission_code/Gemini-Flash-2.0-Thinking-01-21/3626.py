class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def digit_product(num):
            product = 1
            if num == 0:
                return 0
            temp_num = num
            if temp_num < 0:
                temp_num = -temp_num
            while temp_num > 0:
                digit = temp_num % 10
                product *= digit
                temp_num //= 10
            return product

        current_num = n
        while True:
            prod = digit_product(current_num)
            if t == 1:
                return current_num
            if prod % t == 0:
                return current_num
            current_num += 1