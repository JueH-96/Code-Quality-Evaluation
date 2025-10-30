MOD = 10**9 + 7

class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        # We want to maximize (a XOR x) * (b XOR x) for 0<= x < 2^n.
        # Define:
        #   f = a XOR x, g = b XOR x.
        # Notice that if we let u = f then
        #   g = b XOR x = b XOR (a XOR u) = (a XOR b) XOR u.
        # Let c = a XOR b.
        # Then our problem is to choose u in [0, 2^n) to maximize  
        #   F(u) = u * (u XOR c).
        # Because mapping x -> u = a XOR x is a bijection, the maximum is achieved
        # by the choice of u that maximizes u*(u XOR c). 
        # To form u and v = u XOR c bit‐by‐bit (for bits 0..n-1) in such a way to maximize the product,
        # note that u and u XOR c must be as “balanced” as possible.
        #
        # We process bit positions from the most significant (n-1) down to 0.
        # At each bit position i, let c_i be the bit of c at that position.
        # The i‐th bits of u and v = u XOR c are related:
        #   If c_i == 0 => u_i == v_i.
        #   If c_i == 1 => u_i and v_i differ (one is 0 and the other is 1).
        # When c_i==0, to maximize both numbers it is always beneficial to choose 1 at that bit.
        # When c_i==1, we have a choice:
        #   Either set (u_i, v_i) = (0, 1) or (1, 0).
        # To maximize the product u*v (i.e. to keep them as close as possible)
        # we want the two resulting numbers to be balanced.
        # Suppose that after processing higher bits, we have partial values X (for u) and Y (for v).
        # At a bit position where c_i==1, if X < Y then choosing u_i = 1 (so that v_i becomes 0)
        # will help balance the numbers (increasing the smaller one). Otherwise, if X >= Y, we choose u_i = 0.
        #
        # The following greedy algorithm constructs u (and implicitly v = u XOR c) bit by bit:
        #
        #   Let c = a XOR b.
        #   Initialize X = 0, Y = 0.
        #   For i from n-1 downto 0:
        #       Let bit = 1 << i.
        #       If the i-th bit of c is 0:
        #           Set u_i = 1   (so both X and Y get that bit)
        #           X = X*2 + 1, Y = Y*2 + 1.
        #       Else:  # c_i == 1, so bits differ
        #           If X < Y:
        #               Set u_i = 1, so v_i = 0.
        #               X = X*2 + 1, Y = Y*2 + 0.
        #           Else:
        #               Set u_i = 0, so v_i = 1.
        #               X = X*2 + 0, Y = Y*2 + 1.
        #
        # At the end, X = (a XOR x) and Y = (b XOR x) for the x that achieves maximum product.
        # We then return (X * Y) mod (10^9+7).
        
        c = a ^ b
        X, Y = 0, 0  # These will be built bit by bit for u and for u XOR c respectively.
        # Process from the most significant bit (n-1) to least significant (0)
        for i in range(n-1, -1, -1):
            bit_val = 1 << i
            # Check bit i of c:
            if c & bit_val == 0:
                # When c_i == 0, u_i == (u XOR c)_i.
                # To maximize both numbers, choose bit 1.
                X = (X << 1) | 1
                Y = (Y << 1) | 1
            else:
                # c_i == 1 so u_i and (u XOR c)_i differ.
                # Choose the configuration that helps balance X and Y.
                if X < Y:
                    # Increase X to balance.
                    X = (X << 1) | 1
                    Y = (Y << 1) | 0
                else:
                    X = (X << 1) | 0
                    Y = (Y << 1) | 1
        
        # The product is X * Y modulo MOD.
        return (X * Y) % MOD

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    print(sol.maximumXorProduct(12, 5, 4))  # Expected output: 98
    # Example 2
    print(sol.maximumXorProduct(6, 7, 5))   # Expected output: 930
    # Example 3
    print(sol.maximumXorProduct(1, 6, 3))   # Expected output: 12