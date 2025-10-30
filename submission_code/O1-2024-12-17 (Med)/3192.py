class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        """
        We want to maximize ( (a ^ x) * (b ^ x) ) for 0 <= x < 2^n, 
        and return the result modulo 10^9+7.
        
        Key observations:
          1) For bits ≥ n, x has no effect (x < 2^n means x has at most n bits).
             So the top bits of (a^x) and (b^x) are exactly the top bits of a and b.
          2) For bits < n, we control those bits of (a^x) and (b^x) by choosing x.
          3) In particular, (a^x) XOR (b^x) = a XOR b for all x, so for each bit i < n:
               - If a_i == b_i, then (a^x)_i == (b^x)_i.
                 We can choose them both 0 or both 1 by picking x_i appropriately.
               - If a_i != b_i, then (a^x)_i != (b^x)_i.
                 We can choose (1,0) or (0,1) by x_i = 0 or 1.
          4) To maximize the product, it turns out best to set any "same" bits to 1 
             in both (so that both numbers get that bit set).  For "different" bits 
             we must decide whether to give the '1' to (a^x) or to (b^x).  
             We look at which of a or b has larger top half (a>>n vs b>>n). 
             If a>>n > b>>n, we favor making (b^x) bigger.  If b>>n > a>>n, favor (a^x). 
             If they are equal, we distribute these differing bits so that the lower-n-bit
             parts stay as balanced as possible (thus helping the product).
        
        This runs in O(n) and handles up to n=50, a,b<2^50 efficiently.
        """
        MOD = 10**9 + 7
        
        # Separate top and bottom parts of a, b.
        # "hi" are the bits above n; "lo" are the bottom n bits.
        mask = (1 << n) - 1
        a_lo = a & mask
        b_lo = b & mask
        a_hi = a >> n
        b_hi = b >> n
        
        # We'll build A_lo, B_lo for (a^x) and (b^x) in the bottom n bits
        # according to the above logic.
        A_lo = 0
        B_lo = 0
        
        # We'll keep track of the difference D = A_lo - B_lo while assigning bits
        # (only when a_hi == b_hi do we try to keep them balanced).
        D = 0
        
        # Process bits from most significant (n-1) down to 0
        for i in range(n - 1, -1, -1):
            ai = (a_lo >> i) & 1
            bi = (b_lo >> i) & 1
            
            if ai == bi:
                # a_i == b_i -> we can set both bits to 1 (maximizes both numbers)
                # (x_i chosen accordingly, but we needn't store x explicitly).
                # So set bit i in both A_lo and B_lo.
                A_lo |= (1 << i)
                B_lo |= (1 << i)
                # D does not change because we add the same power of 2 to both sides.
            else:
                # a_i != b_i -> exactly one of (a^x)_i, (b^x)_i is 1; the other is 0.
                # Decide based on which top part is larger or if equal, try to balance.
                if a_hi > b_hi:
                    # Favor B_lo to be bigger.
                    B_lo |= (1 << i)
                    D -= (1 << i)
                elif b_hi > a_hi:
                    # Favor A_lo to be bigger.
                    A_lo |= (1 << i)
                    D += (1 << i)
                else:
                    # a_hi == b_hi, use D to keep them balanced.
                    if D >= 0:
                        # A_lo >= B_lo so far -> give the '1' to B_lo
                        B_lo |= (1 << i)
                        D -= (1 << i)
                    else:
                        # B_lo > A_lo so far -> give the '1' to A_lo
                        A_lo |= (1 << i)
                        D += (1 << i)
        
        # Now (a^x) = (a_hi << n) + A_lo, (b^x) = (b_hi << n) + B_lo.
        # Compute the product modulo 10^9+7:
        
        # In Python we can multiply directly then take mod, since Python has 
        # arbitrary precision integers.
        partA = ((a_hi << n) + A_lo) % MOD
        partB = ((b_hi << n) + B_lo) % MOD
        return (partA * partB) % MOD