class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        def is_beautiful(n):
            # Calculate sum and product of digits
            digit_sum = 0
            digit_product = 1
            temp = n
            
            while temp > 0:
                digit = temp % 10
                digit_sum += digit
                digit_product *= digit
                temp //= 10
            
            # Check if product is divisible by sum
            # Need to handle case where sum is 0 (though this shouldn't happen for positive integers)
            if digit_sum == 0:
                return False
            
            return digit_product % digit_sum == 0
        
        # Count beautiful numbers in range
        count = 0
        for num in range(l, r + 1):
            if is_beautiful(num):
                count += 1
        
        return count