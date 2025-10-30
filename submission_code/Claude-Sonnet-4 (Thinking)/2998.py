class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        def is_symmetric(num):
            s = str(num)
            n = len(s)
            
            # Numbers with odd number of digits are never symmetric
            if n % 2 == 1:
                return False
            
            half = n // 2
            first_half_sum = sum(int(digit) for digit in s[:half])
            second_half_sum = sum(int(digit) for digit in s[half:])
            
            return first_half_sum == second_half_sum
        
        count = 0
        for num in range(low, high + 1):
            if is_symmetric(num):
                count += 1
        
        return count