class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7

        # x_ans will store the optimal x value determined bit by bit
        x_ans = 0

        # current_a_xor_x and current_b_xor_x simulate the values
        # (a XOR x_prefix) and (b XOR x_prefix) respectively.
        # They guide the greedy decision for the current bit.
        current_a_xor_x = 0
        current_b_xor_x = 0

        # Iterate from the most significant bit (n-1) down to the least significant bit (0)
        for i in range(n - 1, -1, -1):
            a_bit_i = (a >> i) & 1
            b_bit_i = (b >> i) & 1
            pow2_i = (1 << i)

            # Case 1: The i-th bits of a and b are the same
            if a_bit_i == b_bit_i:
                # To maximize (a XOR x) and (b XOR x), we want A_i and B_i (the i-th bits of A and B) to be 1.
                # If a_i (and b_i) is 0, we need x_i to be 1 to make A_i=1 and B_i=1.
                # If a_i (and b_i) is 1, we need x_i to be 0 to make A_i=1 and B_i=1.
                # So, we only set x_i to 1 if a_i is 0.
                if a_bit_i == 0:
                    x_ans |= pow2_i
                
                # In this case (a_i == b_i), both A and B contribute pow2_i to their values
                # regardless of x_i (because x_i is chosen to make them 1).
                current_a_xor_x |= pow2_i
                current_b_xor_x |= pow2_i
            
            # Case 2: The i-th bits of a and b are different
            else: # a_bit_i != b_bit_i
                # In this case, A_i and B_i will always be different (one 0, one 1).
                # To maximize the product A * B, we want A and B to be as close to each other as possible.
                # We decide x_i based on which of current_a_xor_x or current_b_xor_x is smaller.
                
                if current_a_xor_x < current_b_xor_x:
                    # current_a_xor_x is smaller, so we want A_i to be 1 and B_i to be 0
                    # A_i = (a_bit_i XOR x_i) = 1  => x_i = 1 - a_bit_i
                    # B_i = (b_bit_i XOR x_i) = 0  => x_i = b_bit_i
                    # Since a_bit_i != b_bit_i, (1 - a_bit_i) is indeed equal to b_bit_i.
                    # So, we set x_i to b_bit_i.
                    x_ans |= (b_bit_i << i)
                    
                    # Update current_a_xor_x to reflect that A received the 1 at this bit position.
                    current_a_xor_x |= pow2_i
                else: # current_a_xor_x >= current_b_xor_x
                    # current_b_xor_x is smaller or equal, so we want B_i to be 1 and A_i to be 0
                    # B_i = (b_bit_i XOR x_i) = 1  => x_i = 1 - b_bit_i
                    # A_i = (a_bit_i XOR x_i) = 0  => x_i = a_bit_i
                    # Since a_bit_i != b_bit_i, (1 - b_bit_i) is indeed equal to a_bit_i.
                    # So, we set x_i to a_bit_i.
                    x_ans |= (a_bit_i << i)
                    
                    # Update current_b_xor_x to reflect that B received the 1 at this bit position.
                    current_b_xor_x |= pow2_i
        
        # After iterating through all n bits, x_ans holds the optimal x value.
        # Calculate the final A and B values using the determined x_ans.
        final_a_val = a ^ x_ans
        final_b_val = b ^ x_ans
        
        # Return the product modulo 10^9 + 7.
        return (final_a_val * final_b_val) % MOD