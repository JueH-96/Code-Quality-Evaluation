import bisect
from typing import List

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        """
        Calculates the maximum sum of a balanced subsequence.

        The method transforms the problem based on the balance condition
        `nums[i_j] - nums[i_j-1] >= i_j - i_j-1`, which can be rewritten as
        `nums[i_j] - i_j >= nums[i_{j-1}} - i_{j-1}`. By defining `b[i] = nums[i] - i`,
        the problem becomes finding a non-decreasing subsequence in `b` that
        maximizes the sum of corresponding `nums` elements.

        This is solved using dynamic programming with optimization. A Fenwick tree
        (or Binary Indexed Tree) is used to efficiently find the maximum sum of
        a valid subsequence that can precede the current element. Coordinate
        compression is applied to the `b` values to handle their potentially large
        and sparse range.
        """
        n = len(nums)
        
        # Define b[i] = nums[i] - i. The problem is now to find a subsequence
        # with indices i_0 < i_1 < ... < i_k-1 s.t. b[i_0] <= b[i_1] <= ... <= b[i_k-1]
        # that maximizes sum(nums[i_j]).
        b = [nums[i] - i for i in range(n)]
        
        # Coordinate compress the `b` array as its values can be large.
        coords = sorted(list(set(b)))
        m = len(coords)
        
        # Fenwick Tree for prefix maximums. It will store max DP values.
        # Initialize with a neutral element for 'max' since sums can be negative.
        NEUTRAL = -float('inf')
        bit = [NEUTRAL] * (m + 1)

        def update(idx, val):
            """Updates the BIT with a new max value at a given 1-based index."""
            while idx <= m:
                bit[idx] = max(bit[idx], val)
                idx += idx & -idx
        
        def query(idx):
            """Queries the prefix maximum up to a given 1-based index."""
            max_val = NEUTRAL
            while idx > 0:
                max_val = max(max_val, bit[idx])
                idx -= idx & -idx
            return max_val

        max_total_sum = -float('inf')

        for i in range(n):
            # Find the rank of b[i]. `bisect_left` gives a 0-based index.
            # Add 1 for 1-based BIT indexing.
            rank = bisect.bisect_left(coords, b[i]) + 1
            
            # Query for the maximum sum of a valid subsequence ending before `i`
            # with a `b` value less than or equal to the current `b[i]`.
            prev_max_sum = query(rank)
            
            # A subsequence of length 1 is always valid. If all previous valid
            # subsequences have a negative sum, we start a new one from nums[i].
            # `max(0, prev_max_sum)` achieves this.
            current_sum = nums[i] + max(0, prev_max_sum)
            
            # Update the BIT with the new sum for this rank.
            update(rank, current_sum)
            
            # The final answer is the maximum sum found across all subsequences.
            max_total_sum = max(max_total_sum, current_sum)
            
        return int(max_total_sum)