class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        
        # Build ax = a XOR x and bx = b XOR x bit by bit
        ax = 0
        bx = 0
        
        # Process all bits from position 49 down to 0
        # (since a, b < 2^50, they have at most 50 bits)
        for i in range(49, -1, -1):
            if i >= n:
                # Bits at positions >= n are unaffected by x
                if (a >> i) & 1:
                    ax |= (1 << i)
                if (b >> i) & 1:
                    bx |= (1 << i)
            else:
                # For bits at positions < n, we can control them via x
                bit_a = (a >> i) & 1
                bit_b = (b >> i) & 1
                
                if bit_a == bit_b:
                    # Both have the same bit value
                    # We can make both results have bit i set to 1
                    ax |= (1 << i)
                    bx |= (1 << i)
                else:
                    # Different bit values - exactly one result will have this bit set
                    # Give it to the smaller value to balance them
                    if ax <= bx:
                        ax |= (1 << i)
                    else:
                        bx |= (1 << i)
        
        return (ax * bx) % MOD