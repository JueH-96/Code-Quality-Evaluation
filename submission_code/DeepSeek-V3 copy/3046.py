class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        min_ops = float('inf')
        
        # Iterate over all possible pairs of digits that can form a number divisible by 25
        # The last two digits must be one of the following: 00, 25, 50, 75
        for i in range(n):
            for j in range(i+1, n):
                # Check if the last two digits form a number divisible by 25
                if (int(num[i]) * 10 + int(num[j])) % 25 == 0:
                    # Calculate the number of deletions needed
                    # All digits after j must be deleted
                    ops = (n - 1 - j) + (j - 1 - i)
                    if ops < min_ops:
                        min_ops = ops
        
        # Also consider the case where the number is reduced to 0
        # This happens if we delete all digits except one '0'
        # Find the last '0' in the string
        last_zero = -1
        for i in range(n):
            if num[i] == '0':
                last_zero = i
        if last_zero != -1:
            # Delete all digits after the last '0'
            ops = n - 1 - last_zero
            if ops < min_ops:
                min_ops = ops
        
        # If no valid pair is found and no '0' is present, we need to delete all digits to make it 0
        if min_ops == float('inf'):
            min_ops = n
        
        return min_ops