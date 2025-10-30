class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        def is_beautiful(num):
            if num < 10:  # Single-digit numbers are beautiful
                return True
            
            # Process num digit by digit
            temp_num = num
            digit_sum = 0
            digit_product = 1
            
            while temp_num > 0:
                digit = temp_num % 10
                digit_sum += digit
                digit_product *= digit
                temp_num //= 10
            
            return digit_product % digit_sum == 0
        
        count = 0
        for num in range(l, r + 1):
            if is_beautiful(num):
                count += 1
        
        return count