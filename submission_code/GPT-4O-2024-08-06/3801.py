class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        def is_beautiful(num):
            digits = [int(d) for d in str(num)]
            digit_sum = sum(digits)
            digit_product = 1
            for d in digits:
                digit_product *= d
            # A number is beautiful if the product of its digits is divisible by the sum of its digits
            return digit_sum != 0 and digit_product % digit_sum == 0

        count = 0
        for num in range(l, r + 1):
            if is_beautiful(num):
                count += 1

        return count