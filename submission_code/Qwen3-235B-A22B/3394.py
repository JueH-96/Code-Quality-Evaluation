class Solution:
    def minEnd(self, n: int, x: int) -> int:
        def compute_allowed_a(mask, k):
            allowed_a = 0
            bit_pos = 0
            while k:
                # Find the next bit position in mask that is free (0)
                while (mask & (1 << bit_pos)) != 0:
                    bit_pos += 1
                # Check if current bit of k is set
                if k & 1:
                    allowed_a += 1 << bit_pos
                # Shift k and increment bit_pos
                k >>= 1
                bit_pos += 1
            return allowed_a
        
        k = n - 1
        a = compute_allowed_a(x, k)
        return x + a