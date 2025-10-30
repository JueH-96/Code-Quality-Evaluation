class Solution:
    def smallestNumber(self, n: int) -> int:
        # Start with the given number n
        x = n
        
        # Keep incrementing x until its binary representation contains only set bits
        while (x & (x + 1)) != 0:
            x += 1
        
        return x