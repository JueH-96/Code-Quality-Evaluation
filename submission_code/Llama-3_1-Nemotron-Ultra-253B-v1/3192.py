class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        x = 0
        for i in range(n-1, -1, -1):
            mask = 1 << i
            x_candidate = x | mask
            a_xor = a ^ x_candidate
            b_xor = b ^ x_candidate
            rem_mask = (1 << i) - 1
            max_rem_candidate = ((a_xor & rem_mask) | rem_mask) * ((b_xor & rem_mask) | rem_mask)
            val_candidate = (a_xor & ~rem_mask) * (b_xor & ~rem_mask) + max_rem_candidate
            
            a_xor_current = a ^ x
            b_xor_current = b ^ x
            max_rem_current = ((a_xor_current & rem_mask) | rem_mask) * ((b_xor_current & rem_mask) | rem_mask)
            val_current = (a_xor_current & ~rem_mask) * (b_xor_current & ~rem_mask) + max_rem_current
            
            if val_candidate > val_current:
                x = x_candidate
        return ((a ^ x) * (b ^ x)) % MOD