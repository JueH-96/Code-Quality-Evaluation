class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        
        # For bits at position >= n, we cannot change them
        # Extract the unchangeable parts
        a_high = a >> n << n
        b_high = b >> n << n
        
        # Extract the changeable parts (bits 0 to n-1)
        a_low = a & ((1 << n) - 1)
        b_low = b & ((1 << n) - 1)
        
        # For each bit position from n-1 down to 0
        for i in range(n - 1, -1, -1):
            bit_mask = 1 << i
            
            # Check current bit values
            a_bit = (a_low >> i) & 1
            b_bit = (b_low >> i) & 1
            
            # If bits are different, we want to set the bit in x such that
            # it makes the smaller number larger
            if a_bit != b_bit:
                # Current values after XOR with 0
                curr_a = a_high | a_low
                curr_b = b_high | b_low
                
                if curr_a < curr_b:
                    # Make a larger by setting this bit
                    if a_bit == 0:
                        a_low |= bit_mask
                        b_low &= ~bit_mask
                else:
                    # Make b larger by setting this bit
                    if b_bit == 0:
                        b_low |= bit_mask
                        a_low &= ~bit_mask
            else:
                # If bits are same, setting the bit in x will flip both
                # We want to set both to 1 for maximum value
                if a_bit == 0:  # Both are 0, set them to 1
                    a_low |= bit_mask
                    b_low |= bit_mask
        
        # Calculate final values
        final_a = (a_high | a_low) % MOD
        final_b = (b_high | b_low) % MOD
        
        return (final_a * final_b) % MOD