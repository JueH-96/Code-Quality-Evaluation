class Solution:
    def smallestNumber(self, n: int) -> int:
        # Compute the number with all 1's that has the same bit length as n
        all_ones = (1 << n.bit_length()) - 1
        
        # If n already has all 1's, return n
        if n == all_ones:
            return n
        
        # Otherwise, return the smallest number with all 1's that is >= n
        return all_ones if all_ones >= n else (1 << (n.bit_length() + 1)) - 1