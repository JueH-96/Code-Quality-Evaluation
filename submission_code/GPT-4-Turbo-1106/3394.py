class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # If n is 1, the array can only contain x itself.
        if n == 1:
            return x
        
        # Calculate the highest bit (most significant bit) of x.
        highest_bit = 1
        while highest_bit <= x:
            highest_bit <<= 1
        highest_bit >>= 1
        
        # If n is 2, the array can be [x, x + highest_bit].
        if n == 2:
            return x + highest_bit
        
        # For n > 2, the array can be [x, x + 1, ..., x + n - 1].
        # The last element will be x + n - 1.
        return x + n - 1