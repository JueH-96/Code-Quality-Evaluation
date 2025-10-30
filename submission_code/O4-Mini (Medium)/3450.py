class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # The ball moves back and forth with a full cycle length of 2*(n-1)
        period = 2 * (n - 1)
        # Position within one cycle
        p = k % period
        # If within the forward pass
        if p <= n - 1:
            return p
        # Otherwise it's on the backward pass
        return period - p