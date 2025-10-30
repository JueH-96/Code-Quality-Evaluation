class Solution:
    def minOperations(self, k: int) -> int:
        # We start with nums = [1]. We can:
        # 1) increment any element by 1 (cost = 1 op, increases total sum by 1)
        # 2) duplicate any element (cost = 1 op, increases total sum by the value of that element)
        #
        # Strategy:
        # Suppose we first raise one element from 1 to x (cost = x - 1 increments),
        # then duplicate that element until the total sum is >= k.
        # After raising to x, each duplicate adds x to the sum.
        # We need at least ceil(k / x) total copies of value x,
        # i.e. we need duplicates = ceil(k/x) - 1.
        # Total ops = (x - 1) + (ceil(k/x) - 1) = x + ceil(k/x) - 2.
        #
        # We just try all x from 1 to k (it's up to 1e5, so this is fast enough)
        # and take the minimum.
        
        # Helper for ceiling division
        def ceil_div(a: int, b: int) -> int:
            return (a + b - 1) // b
        
        best = float('inf')
        for x in range(1, k + 1):
            # How many copies of x we need so that x * copies >= k?
            copies = ceil_div(k, x)
            # ops to increment 1 -> x is (x - 1), ops to duplicate (copies - 1) times
            ops = (x - 1) + (copies - 1)
            if ops < best:
                best = ops
        return best