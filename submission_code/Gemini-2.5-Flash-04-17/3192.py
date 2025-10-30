class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7

        # Initialize the final values of A = (a XOR x) and B = (b XOR x) that we are trying to build bit by bit.
        final_A = 0
        final_B = 0

        # Iterate from the most significant relevant bit down to the least significant bit (0).
        # The maximum value of a or b is less than 2^50, so we consider bits up to 49.
        # n is at most 50. If n=50, x can have bits 0..49. If n=0, x can only be 0.
        # We iterate from bit 50 down to 0 to cover all cases. If i=50, the bit value is 2^50.
        # Since a, b < 2^50, bits of a and b at position 50 are 0.
        # If n=50, x can have bits up to 49. If n < 50, x can have bits up to n-1.
        # Bits in x at position i >= n must be 0.

        for i in range(50, -1, -1):
            a_bit = (a >> i) & 1
            b_bit = (b >> i) & 1
            power_of_2 = (1 << i)

            if i >= n:
                # When i >= n, the i-th bit of x must be 0.
                # So, the i-th bit of A is a_bit XOR 0 = a_bit.
                # And the i-th bit of B is b_bit XOR 0 = b_bit.
                # These bits are fixed. Add their contribution to the final values.
                final_A += a_bit * power_of_2
                final_B += b_bit * power_of_2
            else: # i < n, we can choose the i-th bit of x (0 or 1)
                if a_bit == b_bit:
                    # If a_i == b_i, choosing x_i = 0 gives (A_i, B_i) = (a_bit, b_bit).
                    # Choosing x_i = 1 gives (A_i, B_i) = (1 - a_bit, 1 - b_bit).
                    # If a_bit = b_bit = 0, options for (A_i, B_i) are (0,0) [x_i=0] or (1,1) [x_i=1].
                    # If a_bit = b_bit = 1, options for (A_i, B_i) are (1,1) [x_i=0] or (0,0) [x_i=1].
                    # In both cases (a_i == b_i), we can achieve (A_i, B_i) = (1, 1) or (0, 0) by choosing x_i appropriately (x_i = 1 - a_bit for (1,1), x_i = a_bit for (0,0)).
                    # To maximize the product, we want A and B to be larger. Making A_i=1 and B_i=1 adds power_of_2 to both, increasing their magnitudes while keeping their difference unchanged. This is generally better for maximizing the product.
                    # So, we greedily choose (A_i, B_i) = (1, 1).
                    final_A += power_of_2
                    final_B += power_of_2
                else: # a_bit != b_bit
                    # If a_i != b_i, choosing x_i = 0 gives (A_i, B_i) = (a_bit, b_bit).
                    # Choosing x_i = 1 gives (A_i, B_i) = (1 - a_bit, 1 - b_bit).
                    # Since a_i != b_i, 1 - a_bit = b_bit and 1 - b_bit = a_bit.
                    # So, options for (A_i, B_i) are (a_bit, b_bit) or (b_bit, a_bit).
                    # Exactly one of a_bit or b_bit is 1, the other is 0.
                    # This means we can achieve (A_i, B_i) = (1, 0) or (0, 1) by choosing x_i appropriately (x_i = a_bit for (1,0), x_i = b_bit for (0,1)).
                    # We want to add power_of_2 to either final_A or final_B. To maximize the product, we should add it to the currently smaller value to make them closer.
                    if final_A <= final_B:
                         # Aim for (A_i, B_i) = (1, 0) to increase final_A.
                         final_A += power_of_2
                    else:
                         # Aim for (A_i, B_i) = (0, 1) to increase final_B.
                         final_B += power_of_2

        # The values final_A and final_B now represent the greedily constructed numbers A and B that maximize the product.
        # Calculate the product modulo 10^9 + 7.
        # Python handles arbitrary large integers, so intermediate final_A and final_B are fine.
        # Apply modulo only for the final product.
        result = (final_A % MOD * final_B % MOD) % MOD

        return result