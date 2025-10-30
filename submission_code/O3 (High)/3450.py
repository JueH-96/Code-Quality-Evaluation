class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        """
        The children form a line 0 … n-1.
        The ball moves one step each second and bounces when it reaches an end.
        
        The path is periodic:
            0, 1, 2, …, n-1, n-2, …, 1, 0, 1, …
        The length of one full period is 2*(n-1).
        
        Let step = k % (2*(n-1)) be how far we have moved inside the current
        period. If step is still on the forward part (≤ n-1) the position is
        exactly ‘step’. Otherwise we are on the backward part and the position
        is reflected.
        """
        period = 2 * (n - 1)           # length of one back-and-forth trip
        step = k % period              # position inside current period
        
        if step <= n - 1:              # still moving to the right
            return step
        else:                          # moving back to the left, reflect
            return period - step