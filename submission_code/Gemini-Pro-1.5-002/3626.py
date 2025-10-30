class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        
        def digit_product(num):
            prod = 1
            for digit in str(num):
                prod *= int(digit)
            return prod

        num = n
        while True:
            if digit_product(num) % t == 0:
                return num
            num += 1