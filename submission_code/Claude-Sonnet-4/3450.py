class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # The ball creates a cycle of length 2(n-1)
        # It goes: 0->1->2->...->n-1->n-2->...->1->0 (and repeats)
        cycle_length = 2 * (n - 1)
        
        # Find position within the current cycle
        position_in_cycle = k % cycle_length
        
        # If we're in the first half of the cycle (going right)
        if position_in_cycle <= n - 1:
            return position_in_cycle
        # If we're in the second half of the cycle (going left)
        else:
            # We've gone to n-1 and now coming back
            # position_in_cycle - (n-1) tells us how many steps back from n-1
            return n - 1 - (position_in_cycle - (n - 1))