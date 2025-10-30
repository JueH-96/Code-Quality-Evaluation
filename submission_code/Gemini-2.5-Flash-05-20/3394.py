class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # If n is 1, we only need one number in the array.
        # To satisfy (nums[0] & x) = x and minimize nums[0], nums[0] must be x.
        if n == 1:
            return x
        
        # We need to find the (n-1)-th number in a sequence of integers
        # that are constructed by taking the bits of (n-1) and mapping them
        # to the bit positions where x has a 0.
        # The first number in the sequence (nums[0]) is x.
        # The subsequent numbers nums[i] will be of the form x | b_i,
        # where b_i only has bits set in positions where x has a 0.
        # To minimize nums[n-1], we want to find the smallest b_{n-1}
        # such that b_0=0, b_1, ..., b_{n-1} forms an increasing sequence
        # and x | b_i is strictly increasing.
        
        # n_minus_1 represents the (0-indexed) "count" of numbers we need to generate
        # after nums[0] = x.
        # So, if n=3, n_minus_1 = 2. We need to find b_2.
        n_minus_1 = n - 1
        
        # result_b_part will store the value b_{n-1}.
        result_b_part = 0
        
        # current_dest_bit_idx iterates through all possible bit positions (0, 1, 2, ...)
        # to find available slots for bits from n_minus_1.
        current_dest_bit_idx = 0 
        
        # Loop until all bits of n_minus_1 have been processed.
        # This loop will run roughly log2(n) times.
        while n_minus_1 > 0:
            # Check if the bit at current_dest_bit_idx in x is 0.
            # If x has a 1 at this position, it's not a "free" bit for result_b_part,
            # so we skip this position and move to the next.
            # This effectively finds the next available bit position where x has a 0.
            if not ((x >> current_dest_bit_idx) & 1):
                # If this bit position is free (x has a 0 there):
                # Check the least significant bit (LSB) of n_minus_1.
                # If the LSB of n_minus_1 is 1, then set this bit in result_b_part.
                if (n_minus_1 >> 0) & 1: # Equivalent to n_minus_1 % 2 == 1
                    result_b_part |= (1 << current_dest_bit_idx)
                
                # After considering the LSB of n_minus_1, shift n_minus_1 right by 1.
                # This "consumes" the LSB, preparing for the next bit of n_minus_1
                # in the subsequent iteration where a free bit is found.
                n_minus_1 >>= 1
            
            # Always increment current_dest_bit_idx to move to the next physical bit position
            # for the next check against x.
            current_dest_bit_idx += 1
            
        # The minimum value of nums[n-1] is x ORed with the constructed b_{n-1} part.
        # This ensures all bits of x are present, and the additional bits are chosen
        # minimally from positions where x allows it.
        return x | result_b_part