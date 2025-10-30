class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # If n is 1, the minimum possible value is x itself
        if n == 1:
            return x
        
        # Calculate the minimum possible value of nums[n - 1]
        # The minimum value is obtained by setting all bits of x to 1 up to the nth bit
        min_end = (x | (2 ** n) - 1)
        
        return min_end