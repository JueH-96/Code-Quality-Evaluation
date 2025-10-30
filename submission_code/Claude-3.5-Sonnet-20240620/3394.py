class Solution:
    def minEnd(self, n: int, x: int) -> int:
        if n == 1:
            return x
        
        # Find the highest set bit in x
        highest_bit = x.bit_length() - 1
        
        # Start with the minimum possible value that has all bits of x set
        result = x
        
        # Set additional bits to ensure strictly increasing sequence
        for i in range(n - 1):
            next_bit = highest_bit + i + 1
            result |= (1 << next_bit)
        
        return result