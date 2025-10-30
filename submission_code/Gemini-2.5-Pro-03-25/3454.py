import math # This import is not needed but included in the original template context potentially
from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        """
        Calculates the minimum number of operations to make nums equal to target.

        An operation consists of selecting a subarray nums[l..r] and incrementing
        or decrementing all elements within that subarray by 1.

        The problem can be reframed by considering the difference array
        `diff[i] = target[i] - nums[i]`. The goal is to make the `diff` array
        all zeros using the minimum number of operations. An operation on `nums`
        corresponds to adding +1 or -1 to a subarray `diff[l..r]`.

        Let d[i] = target[i] - nums[i].
        Consider the adjacent differences in the `d` array, including boundary
        conditions. Define `delta[i] = d[i] - d[i-1]` for `i = 0..n`,
        where `d[-1] = 0` (value before the start) and `d[n] = 0` (value after the end).
        So, the delta array is:
        delta = [d[0]-d[-1], d[1]-d[0], ..., d[n-1]-d[n-2], d[n]-d[n-1]]
              = [d[0], d[1]-d[0], ..., d[n-1]-d[n-2], -d[n-1]]
        The sum of elements in the delta array is always 0.

        An operation of adding `k` (+1 or -1) to `diff[l..r]` corresponds to
        adding `k` to `delta[l]` and adding `-k` to `delta[r+1]`.

        We want to transform the initial `delta` array to an array of all zeros
        using the minimum number of these paired (+k, -k) operations. The cost
        of each operation is 1 (since we add/subtract by 1).

        Let P = sum(max(0, delta[i])) for i=0..n.
        Let N = sum(min(0, delta[i])) for i=0..n.
        Since sum(delta[i]) = 0, we have P + N = 0, so P = -N = abs(N).
        P represents the total positive value in the delta array, which needs to
        be moved out or cancelled. abs(N) represents the total magnitude of
        negative value, which needs to be filled or cancelled.

        Each operation (inc or dec) effectively moves 1 unit of value between
        two positions in the delta array. The total amount of positive value that
        needs to be reduced (or moved to negative positions) is P.
        Therefore, the minimum number of operations required is P.

        The algorithm calculates P = sum(max(0, delta[i])) for i from 0 to n.

        Args:
            nums: The starting positive integer array.
            target: The target positive integer array.

        Returns:
            The minimum number of operations required.
        """
        n = len(nums)
        # Constraint: 1 <= n <= 10^5, so n is guaranteed to be at least 1.

        operations: int = 0 # Initialize the total operations count (using 64-bit equivalent in Python)
        prev_diff: int = 0 # Represents diff[i-1], initialized to diff[-1] = 0

        # Iterate through indices 0 to n-1 to calculate contributions from delta[0] to delta[n-1]
        for i in range(n):
            # Calculate diff[i] = target[i] - nums[i]
            current_diff = target[i] - nums[i]

            # Calculate delta[i] = diff[i] - diff[i-1]
            delta_i = current_diff - prev_diff

            # Add the positive part of delta[i] to the total operations count
            # This corresponds to max(0, delta[i])
            if delta_i > 0:
                operations += delta_i

            # Update prev_diff for the next iteration, it now holds diff[i]
            prev_diff = current_diff

        # After the loop, prev_diff holds diff[n-1].
        # Now, calculate the contribution from delta[n] = diff[n] - diff[n-1] = 0 - diff[n-1] = -prev_diff
        delta_n = -prev_diff

        # Add the positive part of delta[n] to the total operations count
        # This corresponds to max(0, delta[n])
        if delta_n > 0:
            operations += delta_n

        # The final result `operations` holds sum(max(0, delta[i])) for i=0..n
        return operations