import math # This import is not strictly necessary as we use bitwise operations, but included for clarity if one were to think of powers.

class Solution:
    def smallestNumber(self, n: int) -> int:
        """
        Finds the smallest number x >= n such that the binary representation of x contains only set bits.

        The numbers whose binary representation contains only set bits are of the form 2^k - 1
        for integer k >= 1. These numbers form the sequence: 1, 3, 7, 15, 31, 63, ...
        (In binary: 1, 11, 111, 1111, 11111, 111111, ...)

        The problem asks for the smallest number 'x' in this sequence such that x >= n.

        Args:
            n: A positive integer (1 <= n <= 1000).

        Returns:
            The smallest number x >= n with all set bits in its binary representation.
        """
        
        # We can find the required number 'x' using two main approaches:
        
        # Approach 1: Iterative Generation
        # Start with the first number in the sequence, 1.
        # Keep generating the next number in the sequence (3, 7, 15, ...) until
        # we find a number that is greater than or equal to n.
        # Since the sequence is strictly increasing, the first number found that
        # meets the condition x >= n will be the smallest such number.

        # num_all_ones = 1  # Start with 2^1 - 1
        # while num_all_ones < n:
        #     # Calculate the next number in the sequence.
        #     # If current is 2^k - 1, next is 2^(k+1) - 1.
        #     # This can be calculated efficiently as (current_num << 1) | 1
        #     num_all_ones = (num_all_ones << 1) | 1
        # return num_all_ones

        # Approach 2: Using Bit Length
        # The number of bits in the binary representation of n gives us a clue.
        # n.bit_length() returns the smallest integer k such that n < 2^k.
        # This also means that 2^(k-1) <= n < 2^k.
        # For example:
        # n=5 (binary "101"), n.bit_length() = 3. This means 2^(3-1) <= 5 < 2^3, or 4 <= 5 < 8.
        # n=8 (binary "1000"), n.bit_length() = 4. This means 2^(4-1) <= 8 < 2^4, or 8 <= 8 < 16.
        
        k = n.bit_length()
        
        # Now consider the number formed by k set bits, which is x_k = 2^k - 1.
        # Since n < 2^k, it follows that n <= 2^k - 1. So, x_k is definitely >= n.
        
        # Is x_k the smallest number in the sequence (1, 3, 7, ...) that is >= n?
        # Consider the preceding number in the sequence, x_{k-1} = 2^(k-1) - 1.
        # From the property of bit_length, we know n >= 2^(k-1) (since n is positive).
        # Therefore, x_{k-1} = 2^(k-1) - 1 < 2^(k-1) <= n.
        # This means x_{k-1} is strictly less than n.
        
        # Since the sequence x_m = 2^m - 1 increases with m, and we've shown
        # x_{k-1} < n and x_k >= n, it follows that x_k is the smallest number
        # in the sequence that is greater than or equal to n.
        
        # We can calculate 2^k - 1 efficiently using bitwise operations.
        # 1 << k computes 2^k.
        result = (1 << k) - 1
        
        return result