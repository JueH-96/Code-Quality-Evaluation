class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        
        # Initialize x
        x = 0
        
        # For bits beyond the first n bits, we can't change x, so those bits of x are 0
        # Therefore, for those bits, (a XOR x) = a and (b XOR x) = b
        mask_high = ~((1 << n) - 1)
        A = a & mask_high
        B = b & mask_high
        
        # For the first n bits, we need to decide the value of x
        for i in range(n-1, -1, -1):
            bit_mask = 1 << i
            bit_a = (a >> i) & 1
            bit_b = (b >> i) & 1
            
            if bit_a == bit_b:
                # Both bits are the same
                if bit_a == 0:
                    # Both a[i] and b[i] are 0, setting x[i] = 1 makes both (a XOR x)[i] and (b XOR x)[i] 1
                    x |= bit_mask
                # Both (a XOR x)[i] and (b XOR x)[i] will be 1
                A |= bit_mask
                B |= bit_mask
            else:
                # a[i] and b[i] are different
                if A < B:
                    # Make (a XOR x)[i] = 1
                    if bit_a == 0:
                        x |= bit_mask
                    A |= bit_mask
                else:
                    # Make (b XOR x)[i] = 1
                    if bit_b == 0:
                        x |= bit_mask
                    B |= bit_mask
        
        # Calculate the final result
        result_a = a ^ x
        result_b = b ^ x
        
        return (result_a * result_b) % MOD