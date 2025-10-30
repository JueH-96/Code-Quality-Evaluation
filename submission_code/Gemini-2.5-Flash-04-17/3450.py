class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # The movement of the ball is periodic.
        # It takes n-1 seconds to go from child 0 to child n-1.
        # It takes another n-1 seconds to go from child n-1 back to child 0.
        # The total length of one full cycle (0 -> n-1 -> 0) is (n-1) + (n-1) = 2 * (n-1) seconds.
        # This is the period of the child holding the ball at any given second k.
        # For example, with n=3: 0 -> 1 -> 2 (2 seconds) -> 1 -> 0 (2 seconds). Period = 4.
        # Children sequence: 0, 1, 2, 1, 0, 1, 2, 1, ...
        
        period = 2 * (n - 1)
        
        # We only need to consider the position within the current cycle.
        # The state after k seconds is the same as the state after k % period seconds.
        effective_k = k % period
        
        # If the effective number of steps is less than n-1, the ball is moving
        # towards the right and has not yet reached the rightmost child (n-1).
        # The positions visited during the initial rightward movement are 0, 1, 2, ..., n-2.
        # At time t (where 0 <= t <= n-2), the ball is at child t.
        # If effective_k is between 0 and n-2 (inclusive), the child is effective_k.
        if effective_k < n - 1:
            return effective_k
        else:
            # If the effective number of steps is n-1 or more, the ball has reached
            # the rightmost child (n-1) or gone past it within the cycle and is moving towards the left.
            # At effective_k = n-1, the ball is at child n-1.
            # For effective_k > n-1, the ball moves left from n-1.
            # The number of steps taken moving left from child n-1 is effective_k - (n - 1).
            # Starting from position n-1, moving left by m steps reaches position (n-1) - m.
            # Here, m = effective_k - (n - 1).
            # The position is (n - 1) - (effective_k - (n - 1))
            #              = n - 1 - effective_k + n - 1
            #              = 2 * (n - 1) - effective_k
            # This formula correctly covers effective_k from n-1 up to 2*(n-1).
            # If effective_k = n-1, result is 2(n-1) - (n-1) = n-1.
            # If effective_k = n, result is 2(n-1) - n = n-2.
            # If effective_k = 2(n-1), result is 2(n-1) - 2(n-1) = 0.
            
            return 2 * (n - 1) - effective_k