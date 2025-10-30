class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # The ball moves back and forth along the line. 
        # Its movement is periodic with a period of 2*(n-1).
        # Calculate the effective steps within one period.
        m = k % (2 * (n - 1))
        # If m is within the forward pass (child 0 to child n-1), the position is m.
        # Otherwise, during the backward pass, it will be reversed.
        return m if m <= n - 1 else 2 * (n - 1) - m