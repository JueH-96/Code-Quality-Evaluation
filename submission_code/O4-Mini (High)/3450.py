class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # The ball movement on [0, n-1] with bounces has period = 2*(n-1).
        period = 2 * (n - 1)
        # Find effective steps within one period
        d = k % period
        # If d is within the forward leg [0, n-1], position is d;
        # otherwise it's on the return leg, at position period - d.
        if d <= n - 1:
            return d
        else:
            return period - d