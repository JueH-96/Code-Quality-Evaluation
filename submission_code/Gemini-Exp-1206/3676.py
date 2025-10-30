class Solution:
    def smallestNumber(self, n: int) -> int:
        binary = bin(n)[2:]
        length = len(binary)
        
        result = (1 << length) - 1
        
        if result >= n:
            return result
        else:
            return (1 << (length + 1)) - 1