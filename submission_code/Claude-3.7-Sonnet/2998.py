class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        
        for num in range(low, high + 1):
            # Convert to string to easily work with digits
            num_str = str(num)
            n = len(num_str)
            
            # Skip numbers with odd number of digits
            if n % 2 != 0:
                continue
            
            # Calculate the sum of first half and second half digits
            half_length = n // 2
            first_half_sum = sum(int(digit) for digit in num_str[:half_length])
            second_half_sum = sum(int(digit) for digit in num_str[half_length:])
            
            # If sums are equal, it's a symmetric integer
            if first_half_sum == second_half_sum:
                count += 1
        
        return count