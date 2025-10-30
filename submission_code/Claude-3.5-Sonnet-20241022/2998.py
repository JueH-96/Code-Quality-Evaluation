class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        def is_symmetric(num):
            # Convert number to string
            num_str = str(num)
            length = len(num_str)
            
            # Numbers with odd digits can't be symmetric
            if length % 2 != 0:
                return False
                
            # Get first half and second half
            mid = length // 2
            first_half = num_str[:mid]
            second_half = num_str[mid:]
            
            # Compare sums of digits
            first_sum = sum(int(d) for d in first_half)
            second_sum = sum(int(d) for d in second_half)
            
            return first_sum == second_sum
            
        count = 0
        for num in range(low, high + 1):
            if is_symmetric(num):
                count += 1
                
        return count