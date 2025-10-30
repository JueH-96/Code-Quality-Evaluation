class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        """
        Calculates the minimum number of operations to make num1 zero.

        An operation consists of subtracting (2^i + num2) from num1.
        After k operations, the condition for num1 to be zero is:
        num1 - k * num2 = 2^i_1 + 2^i_2 + ... + 2^i_k
        
        Let target = num1 - k * num2.
        We need to find the smallest k >= 1 such that `target` can be
        represented as a sum of k powers of 2. This requires:
        1. target >= k (since each 2^i >= 1)
        2. The number of set bits in target (popcount) <= k.
        
        We iterate through k and check these conditions.
        """
        
        # We are looking for the smallest integer k >= 1.
        # Let's iterate through possible values of k. The problem constraints,
        # especially i <= 60, suggest k won't be extremely large. A loop up to
        # 61 is a robust choice.
        for k in range(1, 62):
            target = num1 - k * num2
            
            # The conditions for `target` to be a sum of `k` powers of 2 are:
            # 1. target >= k: The sum of k powers of 2 (each >= 1) is at least k.
            #    This also implies target > 0, as k >= 1.
            # 2. target.bit_count() <= k: The minimum number of powers of 2 that
            #    sum to `target` is its popcount. We can always use more terms,
            #    so k must be at least the popcount.
            
            if target >= k and target.bit_count() <= k:
                return k
        
        # If the loop completes without finding a suitable k, it's impossible.
        return -1