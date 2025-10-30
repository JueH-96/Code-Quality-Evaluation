class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # The ball will be at the same position after every 2*(n-1) seconds
        # because it takes n-1 seconds to reach one end and another n-1 seconds to come back to the starting point.
        k = k % (2 * (n - 1))
        
        # If k is less than n, the ball is moving towards the right for the first time.
        if k < n:
            return k
        # If k is greater or equal to n, the ball has reached the right end and is moving back towards the left.
        else:
            return 2 * (n - 1) - k