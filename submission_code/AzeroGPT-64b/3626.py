class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        if t == 1:
            return n
        
        while True:
            product = 1
            num = n
            while num > 0:
                digit = num % 10
                if digit > 0:
                    product *= digit
                num //= 10
            if product % t == 0:
                return n
            n += 1