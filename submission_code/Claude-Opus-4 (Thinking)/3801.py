class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        count = 0
        for num in range(l, r + 1):
            product = 1
            sum_digits = 0
            n = num
            while n:
                digit = n % 10
                product *= digit
                sum_digits += digit
                n //= 10
            if product % sum_digits == 0:
                count += 1
        return count