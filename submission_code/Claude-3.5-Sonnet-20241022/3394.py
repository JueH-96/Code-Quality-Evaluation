class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # If x has any 0 bits, they must be 0 in all numbers
        # So first number must be exactly x
        curr = x
        
        # For each position in array after first
        for i in range(n-1):
            # Find next number greater than curr that maintains AND = x
            curr += 1
            # Keep incrementing until AND with x equals x
            while (curr & x) != x:
                curr += 1
                
        return curr