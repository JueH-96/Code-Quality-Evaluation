class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # We need to create n numbers where their AND is x
        # The first number should be x itself
        # For subsequent numbers, we increment using only the bit positions where x has 0s
        
        result = x
        n -= 1  # We already have the first number as x
        
        # Find positions where x has 0 bits - these are the positions we can modify
        pos = 0
        while n > 0:
            # Find the next position where x has a 0 bit
            while (x >> pos) & 1:
                pos += 1
            
            # If the current bit of n is 1, set this position in result
            if n & 1:
                result |= (1 << pos)
            
            n >>= 1
            pos += 1
        
        return result