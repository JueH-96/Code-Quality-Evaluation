class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # The ball moves back and forth in a cycle of length 2*(n-1)
        cycle_length = 2 * (n - 1)
        # Find the position in the cycle
        position = k % cycle_length
        if position < n:
            return position
        else:
            return cycle_length - position