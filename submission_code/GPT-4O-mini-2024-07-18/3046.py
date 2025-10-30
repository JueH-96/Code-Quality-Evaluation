class Solution:
    def minimumOperations(self, num: str) -> int:
        # We need to find the minimum number of deletions to make the number special (divisible by 25)
        # A number is divisible by 25 if it ends with 00, 25, 50, or 75
        
        target_endings = ['00', '25', '50', '75']
        min_operations = float('inf')
        
        for ending in target_endings:
            # We will look for the digits of the ending in reverse order
            operations = 0
            j = len(num) - 1
            
            for digit in reversed(ending):
                # Find the digit in num from the end
                while j >= 0 and num[j] != digit:
                    operations += 1
                    j -= 1
                # If we run out of digits in num, we can't form this ending
                if j < 0:
                    operations = float('inf')
                    break
                # Move past the found digit
                j -= 1
            
            # Count remaining digits that need to be deleted
            operations += j + 1  # All digits before the last found digit
            
            min_operations = min(min_operations, operations)
        
        return min_operations