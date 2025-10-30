class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        """
        Finds the maximum value of (a XOR x) * (b XOR x) for 0 <= x < 2^n.

        The approach is to build the two numbers, final_a = (a XOR x) and final_b = (b XOR x),
        greedily from the most significant bit (MSB) to the least significant bit (LSB).
        This greedy strategy works because decisions on higher-order bits have a much
        larger impact on the final value than decisions on lower-order bits.

        We iterate from a sufficiently high bit position (49, since a, b < 2^50) down to 0.

        For each bit position k:
        1. If k >= n: The k-th bit of x must be 0. So, the k-th bits of final_a and final_b
           are fixed to be a_k and b_k respectively.

        2. If k < n: We can choose the k-th bit of x (x_k) to be either 0 or 1.
           - If a_k == b_k: To maximize the product, we make both numbers larger.
             We choose x_k such that final_a_k and final_b_k are both 1.
           - If a_k != b_k: One of final_a_k, final_b_k will be 1 and the other 0.
             To keep the numbers close (and maximize their product), we give the '1' bit
             to the number that is currently smaller based on the bits processed so far.
        """
        MOD = 10**9 + 7

        # final_a and final_b will be constructed greedily.
        final_a = 0
        final_b = 0

        # Iterate from bit 49 down to 0, which covers the bit range for the given constraints.
        for k in range(49, -1, -1):
            a_k = (a >> k) & 1
            b_k = (b >> k) & 1

            # Determine the k-th bit for final_a and final_b
            if k >= n:
                # x_k must be 0, so bits of final_a/b are the same as a/b
                final_a_k = a_k
                final_b_k = b_k
            else:  # k < n, we can choose x_k
                if a_k == b_k:
                    # To maximize, make both bits 1. This adds 2^k to both numbers.
                    final_a_k = 1
                    final_b_k = 1
                else:  # a_k != b_k
                    # One bit will be 1, the other 0.
                    # Give the '1' bit to the smaller of the two numbers constructed so far.
                    if final_a > final_b:
                        # Make final_b larger to bring it closer to final_a
                        final_a_k = 0
                        final_b_k = 1
                    else:
                        # Make final_a larger (or if they are equal)
                        final_a_k = 1
                        final_b_k = 0
            
            # Update final_a and final_b with the chosen bits
            if final_a_k == 1:
                final_a |= (1 << k)
            if final_b_k == 1:
                final_b |= (1 << k)
            
        # Calculate the product modulo MOD
        res_a = final_a % MOD
        res_b = final_b % MOD
        
        return (res_a * res_b) % MOD