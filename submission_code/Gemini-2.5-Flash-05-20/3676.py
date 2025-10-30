class Solution:
    def smallestNumber(self, n: int) -> int:
        # Step 1: Determine the number of bits in the binary representation of n.
        # For example:
        # n = 1 (0b1), num_bits = 1
        # n = 3 (0b11), num_bits = 2
        # n = 5 (0b101), num_bits = 3
        # n = 10 (0b1010), num_bits = 4
        num_bits = n.bit_length()
        
        # Step 2: Construct the smallest number x >= n that consists only of set bits.
        # A number with 'k' set bits (all 1s) is equivalent to (2^k) - 1.
        # Since 'n' requires 'num_bits' to represent, the smallest number with all 1s
        # that has 'num_bits' is (2^num_bits) - 1.
        # This value will always be greater than or equal to 'n' because 'n' itself
        # is less than or equal to the maximum value representable by 'num_bits' (which is 2^num_bits - 1).
        # Any number with fewer than 'num_bits' all-set bits (e.g., 2^(num_bits-1) - 1)
        # would be strictly less than 'n' (as n >= 2^(num_bits-1)).
        
        # Calculate (2^num_bits) - 1 using bit shift.
        # (1 << num_bits) is equivalent to 2^num_bits.
        result = (1 << num_bits) - 1
        
        return result