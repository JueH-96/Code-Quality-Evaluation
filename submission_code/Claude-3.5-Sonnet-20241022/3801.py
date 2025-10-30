class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        def is_beautiful(num):
            # Convert number to string to process digits
            digits = str(num)
            
            # Calculate product of digits
            product = 1
            # Calculate sum of digits
            digit_sum = 0
            
            for digit in digits:
                d = int(digit)
                product *= d
                digit_sum += d
            
            # Check if product is divisible by sum
            # Handle case where sum is 0
            return digit_sum != 0 and product % digit_sum == 0
        
        # Count beautiful numbers in range
        count = 0
        for num in range(l, r + 1):
            if is_beautiful(num):
                count += 1
                
        return count