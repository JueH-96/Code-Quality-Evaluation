class Solution:
    def minChanges(self, n: int, k: int) -> int:
        """
        We can only turn 1-bits of n into 0-bits.
        
        1.  k must not contain a 1-bit where n has a 0-bit.
            This is exactly the condition  (k & ~n) == 0 .
            If it fails, the transformation is impossible.
            
        2.  Every 1-bit that is present in n but **not** in k
            must be cleared.  
            These bits are given by  n & ~k .
            The number of such bits is the minimal number
            of required changes.
        """
        # 1. feasibility check
        if k & ~n:          # k asks for a 1 where n has 0
            return -1

        # 2. how many 1-bits must be cleared?
        to_clear = n & ~k
        # use bit_count() when available (Py ≥ 3.8), otherwise fallback
        return to_clear.bit_count() if hasattr(int, "bit_count") else bin(to_clear).count('1')