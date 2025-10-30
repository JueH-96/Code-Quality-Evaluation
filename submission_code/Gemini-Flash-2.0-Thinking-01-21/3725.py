from collections import deque
from typing import List

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_sum = 0

        # dq_max stores indices i such that nums[i] is decreasing.
        # For a window [L, j], dq_max stores indices d_0, d_1, ..., d_m in increasing order
        # L <= d_0 < d_1 < ... < d_m <= j
        # and nums[d_0] > nums[d_1] > ... > nums[d_m].
        dq_max = deque()

        # dq_min stores indices i such that nums[i] is increasing.
        # For a window [L, j], dq_min stores indices d'_0, d'_1, ..., d'_{m'} in increasing order
        # L <= d'_0 < d'_1 < ... < d'_{m'} <= j
        # and nums[d'_0] < ... < nums[d'_{m'}].
        dq_min = deque()

        for j in range(n):
            # Add current index j to deques while maintaining monotonic properties.
            # For dq_max: remove indices from the right whose values are <= nums[j].
            # This ensures dq_max stores indices p_0 < p_1 < ... where nums[p_i] are strictly decreasing.
            # Using <= handles duplicates: if nums[p] == nums[j] with p < j, p is removed, so j is the rightmost max.
            while dq_max and nums[dq_max[-1]] <= nums[j]:
                dq_max.pop()
            dq_max.append(j)

            # For dq_min: remove indices from the right whose values are >= nums[j].
            # This ensures dq_min stores indices p_0 < p_1 < ... where nums[p_i] are strictly increasing.
            # Using >= handles duplicates: if nums[p] == nums[j] with p < j, p is removed, so j is the rightmost min.
            while dq_min and nums[dq_min[-1]] >= nums[j]:
                dq_min.pop()
            dq_min.append(j)

            # The left boundary for subarrays ending at j with length <= k.
            # A subarray nums[i...j] has length j - i + 1.
            # j - i + 1 <= k => i >= j - k + 1.
            # The left index i must also be non-negative, so i >= 0.
            # Thus, the valid range for i is [max(0, j - k + 1), j].
            L = max(0, j - k + 1)

            # Remove indices from the left of the deques that are outside the current window [L, j].
            # The deque should only contain indices p such that L <= p <= j.
            while dq_max and dq_max[0] < L:
                dq_max.popleft()
            while dq_min and dq_min[0] < L:
                dq_min.popleft()

            # Calculate sum of maximums for all subarrays ending at j with length <= k.
            # These subarrays are nums[i...j] for i in [L, j].
            # Let dq_max indices be d_0, d_1, ..., d_m. L <= d_0 < d_1 < ... < d_m <= j. nums[d_0] > ... > nums[d_m].
            # For i in the range (d_{p-1}, d_p] (with d_{-1} = L - 1), max(nums[i...j]) = nums[d_p].
            # The number of such starting indices i is d_p - d_{p-1}.
            # Sum over p from 0 to m: nums[d_p] * (d_p - (d_{p-1} if p > 0 else L - 1))

            # Sum of maximums ending at j
            sum_max_ending_at_j = 0
            # prev_idx_max will represent d_{p-1} or L-1. Initialize with L-1 (the index before L).
            prev_idx_max = L - 1
            # Iterate through the indices in dq_max (which are d_0, d_1, ..., d_m in increasing order)
            for idx in dq_max:
                 # nums[idx] is the maximum for subarrays starting in (prev_idx_max, idx] and ending at j.
                 # The number of such starting indices i is idx - prev_idx_max.
                 sum_max_ending_at_j += nums[idx] * (idx - prev_idx_max)
                 # Update prev_idx_max to the current index for the next iteration.
                 prev_idx_max = idx

            # Calculate sum of minimums for all subarrays ending at j with length <= k.
            # These subarrays are nums[i...j] for i in [L, j].
            # Let dq_min indices be d'_0, d'_1, ..., d'_{m'}. L <= d'_0 < d'_1 < ... < d'_{m'} <= j. nums[d'_0] < ... < nums[d'_{m'}].
            # For i in the range (d'_{p-1}, d'_p] (with d'_{-1} = L - 1), min(nums[i...j]) = nums[d'_p].
            # The number of such starting indices i is d'_p - d'_{p-1}.
            # Sum over p from 0 to m': nums[d'_p] * (d'_p - (d'_{p-1} if p > 0 else L - 1))

            # Sum of minimums ending at j
            sum_min_ending_at_j = 0
            # prev_idx_min will represent d'_{p-1} or L-1. Initialize with L-1 (the index before L).
            prev_idx_min = L - 1
            # Iterate through the indices in dq_min (which are d'_0, d'_1, ..., d'_{m'} in increasing order)
            for idx in dq_min:
                 # nums[idx] is the minimum for subarrays starting in (prev_idx_min, idx] and ending at j.
                 # The number of such starting indices i is idx - prev_idx_min.
                 sum_min_ending_at_j += nums[idx] * (idx - prev_idx_min)
                 # Update prev_idx_min to the current index for the next iteration.
                 prev_idx_min = idx

            # Add the sums for subarrays ending at j to the total sum
            total_sum += sum_max_ending_at_j + sum_min_ending_at_j

        return total_sum