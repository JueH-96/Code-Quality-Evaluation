from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        """
        Calculates the maximum possible sum of heights of a beautiful configuration of towers.
        A configuration is beautiful if 1 <= heights[i] <= maxHeights[i] for all i, 
        and the heights array forms a mountain array. A mountain array has a peak index p
        such that heights are non-decreasing up to p and non-increasing from p onwards.

        The approach uses a monotonic stack to efficiently calculate the maximum sum
        for configurations peaking at each possible index. It computes two arrays:
        left_sums[i]: The maximum sum of heights for indices 0 to i, assuming i is the peak,
                      satisfying non-decreasing constraint up to i and height limits.
        right_sums[i]: The maximum sum of heights for indices i to n-1, assuming i is the peak,
                       satisfying non-increasing constraint from i and height limits.

        The total maximum sum for a peak at index p is left_sums[p] + right_sums[p] - maxHeights[p].
        The final answer is the maximum of these sums over all possible peak indices p.

        Time Complexity: O(N), where N is the number of towers (length of maxHeights).
                       Each element is pushed onto and popped from the stack at most once during
                       each pass (left-to-right and right-to-left).
        Space Complexity: O(N) for storing the left_sums and right_sums arrays, and for the stack
                        in the worst case.
        """
        n = len(maxHeights)
        
        # Calculate left_sums array using a monotonic stack
        # left_sums[i] stores the sum h[0] + ... + h[i] where h[i] = maxHeights[i]
        # and h[k] = min(h[k+1], maxHeights[k]) for k = i-1 down to 0.
        left_sums = [0] * n
        stack = []  # Stores tuples (height, count) representing height segments
        current_total_sum = 0
        for i in range(n):
            current_val = maxHeights[i]
            current_count = 1 # The current element at index i initially forms a segment of count 1
            
            # Process stack segments that are taller than or equal to the current value.
            # When a smaller element arrives, it potentially caps the heights of previous elements.
            while stack and stack[-1][0] >= current_val:
                prev_h, prev_c = stack.pop()
                # Subtract the contribution of the popped segment from the total sum
                current_total_sum -= prev_h * prev_c
                # The current element effectively extends leftwards over the indices
                # of the popped segment, inheriting their counts.
                current_count += prev_c
            
            # Add the new segment represented by current_val and its aggregated count
            stack.append((current_val, current_count))
            # Add the contribution of this new segment to the total sum
            current_total_sum += current_val * current_count
            
            # Store the calculated sum for the prefix ending at i
            left_sums[i] = current_total_sum

        # Calculate right_sums array using a monotonic stack (similar logic, reversed direction)
        # right_sums[i] stores the sum h[i] + ... + h[n-1] where h[i] = maxHeights[i]
        # and h[k] = min(h[k-1], maxHeights[k]) for k = i+1 up to n-1.
        right_sums = [0] * n
        stack = []  # Reset stack for the right-to-left pass
        current_total_sum = 0
        for i in range(n - 1, -1, -1): # Iterate from right to left
            current_val = maxHeights[i]
            current_count = 1
            
            # Process stack segments taller than or equal to the current value
            while stack and stack[-1][0] >= current_val:
                prev_h, prev_c = stack.pop()
                current_total_sum -= prev_h * prev_c
                current_count += prev_c
            
            # Add the new segment
            stack.append((current_val, current_count))
            current_total_sum += current_val * current_count
            
            # Store the calculated sum for the suffix starting at i
            right_sums[i] = current_total_sum
            
        # Calculate the maximum total sum by considering each index p as a potential peak
        max_sum = 0
        for p in range(n):
            # The sum for a configuration peaking at p is formed by combining the optimal left part ending at p
            # and the optimal right part starting at p. In our construction, the peak height at p is set to maxHeights[p].
            # This value maxHeights[p] is included in both left_sums[p] and right_sums[p],
            # so it must be subtracted once to correct for double counting.
            total_sum_p = left_sums[p] + right_sums[p] - maxHeights[p]
            max_sum = max(max_sum, total_sum_p)
            
        return max_sum