from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        
        # Calculate left_sums
        # left_sums[i] = sum of heights h_0...h_i where h_i = maxHeights[i]
        # and h_j = min(maxHeights[j], h_{j+1}) for j < i.
        # This sequence h_0, ..., h_i is non-decreasing towards the right.
        # The maximum possible heights for this left part, given peak at i, are achieved
        # when h_i = maxHeights[i] and h_j = min(maxHeights[j], h_{j+1}) for j < i.
        # This creates a sequence h_i, h_{i-1}, ..., h_0 that is non-increasing.
        left_sums = [0] * n
        stack = [] # Stores indices j such that maxHeights[j] is increasing.
        # This property helps identify segments [p+1, i] where heights are capped by maxHeights[i].
        
        for i in range(n):
            h = maxHeights[i]
            # While stack is not empty and the height at stack top is >= current height h,
            # pop elements from stack. The segments defined by these popped indices
            # had their heights in the derived sequence (ending at the popped index)
            # capped by the height at the popped index. Now, with peak at i (height h),
            # these segments might have their heights capped by h if h is smaller.
            # By popping >= h, we ensure the remaining stack elements < h.
            while stack and maxHeights[stack[-1]] >= h:
                stack.pop()
            
            # The previous boundary is the index below the current top of stack (or -1).
            prev_idx = stack[-1] if stack else -1
            
            # The sum up to i (left_sums[i]) is the sum up to the previous boundary (left_sums[prev_idx])
            # plus the sum of heights in the segment (prev_idx, i].
            # For any k in (prev_idx, i], maxHeights[k] >= h = maxHeights[i]
            # (because any index j in (prev_idx, i) with maxHeights[j] < h would still be on stack
            # or would have prevented prev_idx from being the top).
            # The sequence h_i, h_{i-1}, ..., h_{prev_idx + 1} is non-increasing, calculated by h_k = min(maxHeights[k], h_{k+1}).
            # Since maxHeights[k] >= h for k in (prev_idx, i), h_k = min(maxHeights[k], h_{k+1}) >= min(h, h_{k+1}).
            # Inductively from i down to prev_idx + 1, we get h_k = h for k in (prev_idx, i].
            # The sum of heights in the segment (prev_idx, i] is h * (i - prev_idx).
            left_sum_prev = left_sums[prev_idx] if prev_idx != -1 else 0
            left_sums[i] = left_sum_prev + h * (i - prev_idx)
            
            stack.append(i)
            
        # Calculate right_sums
        # right_sums[i] = sum of heights h_i...h_{n-1} where h_i = maxHeights[i]
        # and h_k = min(maxHeights[k], h_{k-1}) for k > i.
        # This sequence h_i, ..., h_{n-1} is non-increasing towards the right.
        # The maximum possible heights for this right part, given peak at i, are achieved
        # when h_i = maxHeights[i] and h_k = min(maxHeights[k], h_{k-1}) for k > i.
        # This creates a sequence h_i, h_{i+1}, ..., h_{n-1} that is non-increasing.
        right_sums = [0] * n
        stack = [] # Stores indices k (from right to left) such that maxHeights[k] is increasing.
        
        for i in range(n - 1, -1, -1):
            h = maxHeights[i]
            # While stack is not empty and the height at stack top is >= current height h, pop.
            # Similar logic as left_sums, applied from right to left.
            while stack and maxHeights[stack[-1]] >= h:
                stack.pop()

            # The previous boundary is the index below the current top of stack (or n).
            prev_idx = stack[-1] if stack else n
            
            # The sum from i to n-1 (right_sums[i]) is the sum from the next boundary (right_sums[prev_idx])
            # plus the sum of heights in the segment [i, prev_idx).
            # For any k in [i, prev_idx), maxHeights[k] >= h = maxHeights[i]
            # (because any index k in (i, prev_idx) with maxHeights[k] < h would still be on stack
            # or would have prevented prev_idx from being the top).
            # The sequence h_i, h_{i+1}, ..., h_{prev_idx - 1} is non-increasing, calculated by h_k = min(maxHeights[k], h_{k-1}).
            # Since maxHeights[k] >= h for k in [i, prev_idx), inductively we get h_k = h for k in [i, prev_idx).
            # The sum of heights in the segment [i, prev_idx) is h * (prev_idx - i).
            right_sum_prev = right_sums[prev_idx] if prev_idx != n else 0
            right_sums[i] = right_sum_prev + h * (prev_idx - i)

            stack.append(i)

        # Calculate total sum for each peak i and find the maximum
        max_total_sum = 0
        for i in range(n):
            # The height maxHeights[i] is included in both left_sums[i] and right_sums[i].
            # We need to subtract it once to get the total sum for the configuration
            # with peak at i.
            total_sum_at_peak_i = left_sums[i] + right_sums[i] - maxHeights[i]
            max_total_sum = max(max_total_sum, total_sum_at_peak_i)

        return max_total_sum