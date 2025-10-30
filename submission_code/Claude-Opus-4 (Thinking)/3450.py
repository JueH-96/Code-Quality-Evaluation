class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # One complete cycle takes 2*(n-1) seconds
        cycle_length = 2 * (n - 1)
        
        # Reduce k to within one cycle
        k = k % cycle_length
        
        if k <= n - 1:
            # Ball is moving right from position 0
            return k
        else:
            # Ball has reached the right end and is moving left
            # It's at position (n-1) and has moved (k-(n-1)) steps left
            return 2 * (n - 1) - k