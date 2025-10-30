class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        
        for num in range(low, high + 1):
            # Convert number to string to easily access digits
            s = str(num)
            
            # Check if number has even number of digits
            if len(s) % 2 == 0:
                # Split into two halves
                n = len(s) // 2
                
                # Calculate sum of first half
                first_half_sum = sum(int(digit) for digit in s[:n])
                
                # Calculate sum of second half
                second_half_sum = sum(int(digit) for digit in s[n:])
                
                # Check if symmetric
                if first_half_sum == second_half_sum:
                    count += 1
        
        return count