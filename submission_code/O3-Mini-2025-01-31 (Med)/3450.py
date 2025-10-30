class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # The ball moves with a periodic pattern with period = 2*(n-1)
        period = 2 * (n - 1)
        remainder = k % period

        # During one period, if the ball reaches the right end and then goes back,
        # the position can be computed as a "mirrored" version.
        if remainder <= n - 1:
            return remainder
        else:
            return period - remainder