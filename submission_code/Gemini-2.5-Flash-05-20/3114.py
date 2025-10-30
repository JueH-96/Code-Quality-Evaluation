import collections
from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        
        # left_sums[i] will store the maximum possible sum of heights for towers from 0 to i,
        # assuming tower i is the peak of a non-decreasing sequence (h_0 <= ... <= h_i)
        # and h_k <= maxHeights[k] for all k. Specifically, h_k = min(maxHeights[k], ..., maxHeights[i]).
        left_sums = [0] * n
        stack = [] # Stores (index, effective_height) pairs. Monotonically increasing heights based on current tower.
        current_sum_for_prefix = 0 # This accumulates the sum for the current left segment
        
        for i in range(n):
            cur_h = maxHeights[i]
            
            # While the stack is not empty and the height of the top element in stack
            # (which corresponds to a tower to the left of `i`) is greater than or equal to `cur_h`:
            # Pop these elements. Their actual heights are now capped by `cur_h`.
            # We "rollback" their previous sum contribution based on their former height.
            while stack and stack[-1][1] >= cur_h:
                prev_idx, prev_h_val = stack.pop()
                # The elements in the range (stack[-1][0] + 1) to prev_idx were previously `prev_h_val`.
                # We subtract their old contribution. The length of this segment is `prev_idx - (stack[-1][0] if stack else -1)`.
                current_sum_for_prefix -= prev_h_val * (prev_idx - (stack[-1][0] if stack else -1))
            
            # After popping, `cur_h` is now the effective height for a contiguous segment
            # from (stack[-1][0] + 1) up to `i`. This segment length is `i - (stack[-1][0] if stack else -1)`.
            # Add this new contribution to the current sum.
            span_start_idx = stack[-1][0] if stack else -1
            current_sum_for_prefix += cur_h * (i - span_start_idx)
            
            # Push the current tower's information onto the stack.
            stack.append((i, cur_h))
            left_sums[i] = current_sum_for_prefix
            
        # right_sums[i] will store the maximum possible sum of heights for towers from i to n-1,
        # assuming tower i is the peak of a non-increasing sequence (h_i >= ... >= h_n-1)
        # and h_k <= maxHeights[k] for all k. Specifically, h_k = min(maxHeights[k], ..., maxHeights[i]).
        right_sums = [0] * n
        stack = [] # Stores (index, effective_height) pairs. Monotonically increasing heights (from right to left).
        current_sum_for_suffix = 0 # This accumulates the sum for the current right segment

        # Iterate from right to left
        for i in range(n - 1, -1, -1):
            cur_h = maxHeights[i]

            # While the stack is not empty and the height of the top element in stack
            # (which corresponds to a tower to the right of `i`) is greater than or equal to `cur_h`:
            # Pop these elements. Their actual heights are now capped by `cur_h`.
            # We "rollback" their previous sum contribution based on their former height.
            while stack and stack[-1][1] >= cur_h:
                prev_idx, prev_h_val = stack.pop()
                # The elements in the range prev_idx to (stack[-1][0] - 1) were previously `prev_h_val`.
                # We subtract their old contribution. The length of this segment is `(stack[-1][0] if stack else n) - prev_idx`.
                current_sum_for_suffix -= prev_h_val * ((stack[-1][0] if stack else n) - prev_idx)
            
            # After popping, `cur_h` is now the effective height for a contiguous segment
            # from `i` up to (stack[-1][0] - 1). This segment length is `(stack[-1][0] if stack else n) - i`.
            # Add this new contribution to the current sum.
            span_end_idx = stack[-1][0] if stack else n
            current_sum_for_suffix += cur_h * (span_end_idx - i)
            
            # Push the current tower's information onto the stack.
            stack.append((i, cur_h))
            right_sums[i] = current_sum_for_suffix
            
        max_total_sum = 0
        # Iterate through each possible peak position `i`
        for i in range(n):
            # The total sum for a peak at `i` is the sum of the left part (0 to i) plus the sum of the right part (i to n-1).
            # Note that maxHeights[i] (which is h[i]) is included in both `left_sums[i]` and `right_sums[i]`,
            # so we subtract it once to get the total unique sum.
            max_total_sum = max(max_total_sum, left_sums[i] + right_sums[i] - maxHeights[i])
            
        return max_total_sum