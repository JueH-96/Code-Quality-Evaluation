import math

class Solution:
    def minOperations(self, k: int) -> int:
        # If k is 1, the initial array [1] already satisfies the condition.
        if k == 1:
            return 0
        
        ans = float('inf')
        # Let x = b + 1 (final value of each element)
        #     y = d + 1 (final number of elements)
        # We need   x * y >= k   and want to minimise (x - 1) + (y - 1) = x + y - 2.
        limit = int(math.isqrt(k)) + 1
        for x in range(1, limit + 1):
            y = (k + x - 1) // x           # ceil(k / x)
            ans = min(ans, x + y - 2)      # subtract 2 because we counted the initial 1 twice
        return ans