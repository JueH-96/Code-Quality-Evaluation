class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 1000000007
        
        # Get the bits beyond n-th position
        mask = (1 << 50) - (1 << n)
        xa = a & mask
        xb = b & mask
        
        # For positions we can modify (0 to n-1)
        mask = (1 << n) - 1
        remaining_a = a & mask
        remaining_b = b & mask
        
        x = 0
        # Process each bit from n-1 down to 0
        for i in range(n-1, -1, -1):
            bit = 1 << i
            
            # Get current bits at position i
            bit_a = remaining_a & bit
            bit_b = remaining_b & bit
            
            # If bits are different, set x's bit to make smaller number larger
            if (bit_a and not bit_b) or (not bit_a and bit_b):
                if ((xa << 1) + (remaining_a ^ bit)) * ((xb << 1) + (remaining_b ^ bit)) > \
                   ((xa << 1) + remaining_a) * ((xb << 1) + remaining_b):
                    x |= bit
            # If bits are same, set x's bit to make both numbers larger
            else:
                x |= bit
        
        # Calculate final products
        prod_a = (a ^ x) % MOD
        prod_b = (b ^ x) % MOD
        
        return (prod_a * prod_b) % MOD