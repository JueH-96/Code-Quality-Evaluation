class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        
        for num in range(low, high + 1):
            # Convert number to string to easily access digits
            num_str = str(num)
            num_digits = len(num_str)
            
            # Skip if odd number of digits
            if num_digits % 2 != 0:
                continue
            
            # Calculate the midpoint
            n = num_digits // 2
            
            # Sum the first n digits
            first_half_sum = sum(int(digit) for digit in num_str[:n])
            
            # Sum the last n digits
            second_half_sum = sum(int(digit) for digit in num_str[n:])
            
            # Check if symmetric
            if first_half_sum == second_half_sum:
                count += 1
        
        return count