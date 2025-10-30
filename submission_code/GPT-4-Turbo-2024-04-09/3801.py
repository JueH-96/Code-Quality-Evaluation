class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        def is_beautiful(n):
            digits = [int(d) for d in str(n)]
            product = 1
            sum_digits = 0
            for d in digits:
                product *= d
                sum_digits += d
            return product % sum_digits == 0
        
        count = 0
        for num in range(l, r + 1):
            if is_beautiful(num):
                count += 1
        return count