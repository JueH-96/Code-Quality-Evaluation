class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        def is_beautiful(n):
            product = 1
            digit_sum = 0
            temp = n
            while temp > 0:
                digit = temp % 10
                product *= digit
                digit_sum += digit
                temp //= 10
            
            return product % digit_sum == 0
        
        count = 0
        for i in range(l, r + 1):
            if is_beautiful(i):
                count += 1
        return count