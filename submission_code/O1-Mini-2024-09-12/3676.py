class Solution:
    def smallestNumber(self, n: int) -> int:
        """
        Returns the smallest number x >= n such that the binary representation of x contains only set bits (all 1's).

        :param n: A positive integer (1 <= n <= 1000)
        :return: The smallest integer x >= n with all bits set to 1 in its binary representation
        """
        # Calculate the bit length of n
        bit_length = n.bit_length()
        
        # Compute the number with all bits set to 1 for the current bit length
        all_ones = (1 << bit_length) - 1
        
        # If this number is greater than or equal to n, return it
        if all_ones >= n:
            return all_ones
        else:
            # Otherwise, compute the next number with all bits set to 1
            return (1 << (bit_length + 1)) - 1