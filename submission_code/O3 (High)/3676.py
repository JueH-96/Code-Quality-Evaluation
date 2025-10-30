class Solution:
    def smallestNumber(self, n: int) -> int:
        """
        Returns the smallest integer x ≥ n whose binary representation
        consists exclusively of set bits (i.e. is of the form 2^k − 1).
        """
        # Start with the smallest all-ones number: 1 (binary "1")
        x = 1
        
        # Repeatedly extend the sequence of 1-bits until x ≥ n.
        # Each step left-shifts and adds 1, producing:
        # 1 -> 3 -> 7 -> 15 -> ...
        while x < n:
            x = (x << 1) | 1   # equivalent to x = 2*x + 1
        
        return x