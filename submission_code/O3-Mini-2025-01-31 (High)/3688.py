from typing import List
import math
from collections import defaultdict

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        # Compute the global maximum subarray sum using Kadane's algorithm,
        # which corresponds to the “no removal” option.
        global_max = nums[0]
        curr = nums[0]
        for i in range(1, n):
            curr = max(nums[i], curr + nums[i])
            global_max = max(global_max, curr)
        
        # We'll build a segment tree that supports maximum subarray sum queries.
        # Each node holds a 4-tuple: (total, max_prefix, max_suffix, max_subarray)
        # For a single element a, its node is (a, a, a, a).
        
        # Define the combine operation for two segments.
        def combine(left, right):
            total = left[0] + right[0]
            max_prefix = max(left[1], left[0] + right[1])
            max_suffix = max(right[2], right[0] + left[2])
            max_subarray = max(left[3], right[3], left[2] + right[1])
            return (total, max_prefix, max_suffix, max_subarray)
        
        # Define identity element.
        ide = (0, float('-inf'), float('-inf'), float('-inf'))
        
        size = n
        seg = [None] * (2 * size)
        # Build the leaves.
        for i in range(n):
            seg[size + i] = (nums[i], nums[i], nums[i], nums[i])
        # Build the tree upwards.
        for i in range(size - 1, 0, -1):
            seg[i] = combine(seg[2 * i], seg[2 * i + 1])
        
        # Define the query function over [l, r) using the iterative segment tree.
        def query(l, r):
            res_left = ide
            res_right = ide
            l += size
            r += size
            while l < r:
                if l & 1:
                    res_left = combine(res_left, seg[l])
                    l += 1
                if r & 1:
                    r -= 1
                    res_right = combine(seg[r], res_right)
                l //= 2
                r //= 2
            return combine(res_left, res_right)
        
        # Build a dictionary mapping each value x to the list of indices at which it appears.
        pos_dict = defaultdict(list)
        for i, val in enumerate(nums):
            pos_dict[val].append(i)
        
        best_candidate = float('-inf')
        # Try removal of each candidate value x (removing ALL occurrences of x)
        # but only if not all elements equal x.
        for x, positions in pos_dict.items():
            if len(positions) == n:
                # Removing x would empty the array. Skip.
                continue
            
            # The new array (after removal of x) is the concatenation
            # of intervals (segments) between occurrences of x.
            intervals = []
            # Interval before the first occurrence.
            if positions[0] > 0:
                intervals.append((0, positions[0] - 1))
            # Intervals between occurrences.
            for i in range(1, len(positions)):
                start = positions[i - 1] + 1
                end = positions[i] - 1
                if start <= end:
                    intervals.append((start, end))
            # Interval after the last occurrence.
            if positions[-1] < n - 1:
                intervals.append((positions[-1] + 1, n - 1))
            
            if not intervals:
                continue  # This candidate removal would lead to an empty array.
            
            # Combine the segments together using our segment tree queries.
            # (We must "merge" the segments in order so that the contiguous connection
            #  is taken into account.)
            candidate_node = None
            for (l, r) in intervals:
                # our query function uses half-open intervals so we query [l, r+1)
                node = query(l, r + 1)
                if candidate_node is None:
                    candidate_node = node
                else:
                    candidate_node = combine(candidate_node, node)
            best_candidate = max(best_candidate, candidate_node[3])
        
        return max(global_max, best_candidate)