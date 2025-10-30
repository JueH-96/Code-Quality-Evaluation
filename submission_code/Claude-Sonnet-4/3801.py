class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        def is_beautiful(n):
            digits = []
            temp = n
            while temp > 0:
                digits.append(temp % 10)
                temp //= 10
            
            product = 1
            digit_sum = 0
            
            for digit in digits:
                product *= digit
                digit_sum += digit
            
            return product % digit_sum == 0
        
        count = 0
        for num in range(l, r + 1):
            if is_beautiful(num):
                count += 1
                
        return count