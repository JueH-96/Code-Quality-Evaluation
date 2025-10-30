class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        import math
        
        # We want to find the smallest k >= 1 such that:
        #   S = num1 - k*num2 >= 0
        #   popcount(S) <= k
        #   S >= k   (because with k additions of powers of 2 (all >= 1), the smallest sum is k)
        #
        # If no such k up to a reasonable limit is found, return -1.
        # Why up to 61? Because i ranges up to 60, so in principle we could use i=0..60
        # and have at most 61 bits set in total if we needed each distinct bit once.
        # Hence, we only need to check up to k=61.
        
        def popcount(x: int) -> int:
            # A quick way to count set bits in Python:
            return bin(x).count('1')
        
        for k in range(1, 62):
            S = num1 - k*num2
            if S < 0:
                continue
            if S < k:
                continue
            if popcount(S) <= k:
                return k
        
        return -1