class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 1_000_000_007
        
        # We will inspect every bit that can influence the answer.
        # That is all bits below max(n, a.bit_length(), b.bit_length()).
        top = max(n, a.bit_length(), b.bit_length())
        
        u = v = 0          # current values of  (a XOR x)  and  (b XOR x)
        
        # scan bits from the most-significant to the least-significant
        for i in range(top - 1, -1, -1):
            w = 1 << i
            ai = (a >> i) & 1
            bi = (b >> i) & 1
            
            # bits that cannot be changed
            if i >= n:
                xi = 0
            else:
                if ai == bi:                       # both bits are equal
                    # if they are 0 we force them to 1, if they are 1 we keep them 1
                    xi = ai ^ 1
                else:                              # the two bits differ
                    if u > v:                      # give the 1-bit to the smaller prefix (v)
                        xi = bi ^ 1
                    elif v > u:                    # give the 1-bit to the smaller prefix (u)
                        xi = ai ^ 1
                    else:                          # prefixes equal – either choice is fine
                        xi = 0                     # keep the original configuration
            
            # add the decided bit to the running numbers
            u_bit = ai ^ xi
            v_bit = bi ^ xi
            if u_bit:
                u += w
            if v_bit:
                v += w
        
        return (u % MOD) * (v % MOD) % MOD