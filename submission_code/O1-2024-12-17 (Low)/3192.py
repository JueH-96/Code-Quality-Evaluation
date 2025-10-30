class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        """
        We want to choose x in [0, 2^n) to maximize (a ^ x) * (b ^ x), then return
        that product modulo 10^9+7.

        Key idea:
          1) Only the lower n bits of x can differ from zero (since 0 <= x < 2^n).
          2) Let a_up = a >> n, a_low = a & ((1 << n) - 1).  Likewise for b.
             Then (a ^ x) = (a_up << n) + (a_low ^ x), and similarly for b.
          3) We want to pick x's lower n bits cleverly to maximize:
               ( (a_up << n) + (a_low ^ x) ) * ( (b_up << n) + (b_low ^ x) ).
          4) Observe how the lower n bits of (a_low ^ x) and (b_low ^ x) should
             be chosen to get as large a product as possible.
             - Where a_low and b_low have the same bit (both 0 or both 1),
               we can force that bit to become 1 in both factors.
             - Where a_low and b_low differ, that bit can only be 1 in exactly
               one of the two factors.  Distribute these differing bits so that
               (a_low ^ x) and (b_low ^ x) stay as balanced (close in value) as
               possible, which maximizes the product.
          5) Construct the final A = (a_low ^ x) and B = (b_low ^ x) bits as follows:
             - For bits where a_low[i] == b_low[i], set both A[i] and B[i] to 1.
             - For differing bits, assign them in descending (more significant) order
               to whichever of A or B is currently smaller, so they stay balanced.
          6) Reconstruct x from A (since A = a_low ^ x) by x[i] = A[i] ^ a_low[i],
             then compute the product ( (a_up<<n) + A ) * ( (b_up<<n) + B ).

        This approach is efficient (O(n)) and works for n <= 50.
        """
        MOD = 10**9 + 7

        # Split a and b into upper bits and lower n bits
        a_up = a >> n
        b_up = b >> n
        mask = (1 << n) - 1
        a_low = a & mask
        b_low = b & mask

        # Determine which bit positions are the same vs. different in the lower n bits
        same_bits = []
        diff_bits = []
        for i in range(n):
            bit_a = (a_low >> i) & 1
            bit_b = (b_low >> i) & 1
            if bit_a == bit_b:
                same_bits.append(i)
            else:
                diff_bits.append(i)

        # Build the "target" lower bits of (a ^ x) and (b ^ x), call them A and B.
        # Start them at 0, then set bits as per the rules below.
        A = 0
        B = 0

        # 1) For bits that are the same, we want them to be 1 in both A and B.
        #    That guarantees that bit contributes '1' to both factors.
        for i in same_bits:
            A |= (1 << i)
            B |= (1 << i)

        # 2) For bits that differ,
        #    we distribute them so A and B stay close in value (balance).
        #    Sort them descending so we assign the more significant bits first.
        diff_bits.sort(reverse=True)

        for i in diff_bits:
            # Check current values of A and B
            if A <= B:
                # Put this differing bit into A
                A |= (1 << i)
            else:
                # Put this differing bit into B
                B |= (1 << i)

        # Now A = (a_low ^ x), B = (b_low ^ x) for our chosen x.
        # Compute the product using the upper bits.
        val_a = (a_up << n) + A
        val_b = (b_up << n) + B
        product = (val_a % MOD) * (val_b % MOD)
        product %= MOD

        return product