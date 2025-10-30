class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # We look for the smallest k >= 1 such that
        #   S = num1 - k * num2
        # is non-negative, S >= k, and popcount(S) <= k.
        # If no such k <= 60, return -1.
        
        for k in range(1, 61):
            S = num1 - k * num2
            # S must be at least k (we need to choose k powers of two
            # summing to S, which requires S >= k), and non-negative.
            if S < k:
                continue
            # Count 1-bits in S
            if S.bit_count() <= k:
                return k
        return -1