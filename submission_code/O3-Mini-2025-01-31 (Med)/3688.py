from typing import List
import math

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0  # though constraint says non-empty
        
        # ----------------------------
        # We'll build an iterative segment tree for range maximum subarray sum.
        # Each node stores a tuple (total, best_prefix, best_suffix, best_sum)
        # Identity element (for merging an empty interval) is:
        #   total = 0, best_prefix = -infinity, best_suffix = -infinity, best_sum = -infinity
        neginf = float("-inf")
        identity = (0, neginf, neginf, neginf)
        
        # Function to combine two nodes (left then right)
        def combine(left, right):
            total = left[0] + right[0]
            best_prefix = max(left[1], left[0] + right[1])
            best_suffix = max(right[2], right[0] + left[2])
            best_sum = max(left[3], right[3], left[2] + right[1])
            return (total, best_prefix, best_suffix, best_sum)
        
        # Build segment tree. We'll use a complete binary tree size m.
        m = 1
        while m < n:
            m *= 2
        size = 2 * m
        tree = [identity] * size
        
        # Initialize leaves
        for i in range(n):
            # Each leaf value is (value, value, value, value)
            tree[m + i] = (nums[i], nums[i], nums[i], nums[i])
        # The rest remain identity (for indices m+n to m*2-1)
        for i in range(m - 1, 0, -1):
            tree[i] = combine(tree[2 * i], tree[2 * i + 1])
        
        # Define query function: query on [l, r) interval in nums.
        def query(l, r):
            # l and r are 0-indexed; r is exclusive.
            res_left = identity
            res_right = identity
            l += m
            r += m
            while l < r:
                if l & 1:
                    res_left = combine(res_left, tree[l])
                    l += 1
                if r & 1:
                    r -= 1
                    res_right = combine(tree[r], res_right)
                l //= 2
                r //= 2
            return combine(res_left, res_right)
        
        # Compute original maximum subarray sum.
        original_node = query(0, n)
        best = original_node[3]
        
        # Build a dictionary that maps a candidate value to a list of positions (indices) where it occurs.
        # We only consider those removals that leave the array non-empty.
        positions = {}
        for i, v in enumerate(nums):
            if v not in positions:
                positions[v] = []
            positions[v].append(i)
        
        # We try at most one removal operation.
        # We'll iterate over candidates; note that removing a number that is not in the array is the same as no removal.
        # Also, if removal causes the resulting array to be empty, skip.
        # For each candidate x, the filtered array is:
        #   B = [nums[i] for i in range(n) if nums[i] != x]
        # Its maximum subarray sum is computed by "merging" consecutive segments of indices that remain.
        #
        # To get the segments quickly, we know the positions where x occurs (sorted).
        # Then the segments in the original array that remain are:
        #   Segment1: indices [0, pos[0]-1] if pos[0] > 0.
        #   For each i from 1 to len(pos)-1, segment: [pos[i-1]+1, pos[i]-1] if non-empty.
        #   Segment last: [pos[-1]+1, n-1] if pos[-1] < n-1.
        #
        # We then combine these segments (using the same combine that works on segment tree node values)
        # because in the filtered array, these segments become contiguous.
        ans = best  # best so far: either do not remove any value or removal alternatives.
        
        # We'll iterate over candidates.
        for x, idx_list in positions.items():
            # If removal of x results in an empty filtered array, skip.
            if len(idx_list) == n:
                continue
                
            # For many candidates, removal of a positive number might not help.
            # But we still need to check, because sometimes removal could remove an outlier that splits a large sum.
            segments = []
            # first segment: from 0 to idx_list[0]-1, if exists.
            if idx_list[0] > 0:
                l = 0
                r = idx_list[0] - 1
                segments.append( (l, r) )
            # segments between occurrences:
            for i in range(1, len(idx_list)):
                # segment from idx_list[i-1]+1 to idx_list[i]-1
                if idx_list[i-1] + 1 <= idx_list[i] - 1:
                    l = idx_list[i-1] + 1
                    r = idx_list[i] - 1
                    segments.append( (l, r) )
            # last segment:
            if idx_list[-1] < n - 1:
                l = idx_list[-1] + 1
                r = n - 1
                segments.append( (l, r) )
            
            if not segments:
                # nothing remains (all numbers are x) so skip.
                continue
            
            # Merge the segments:
            merged = None
            for (l, r) in segments:
                node = query(l, r + 1)  # r+1 because query is exclusive at right bound.
                if merged is None:
                    merged = node
                else:
                    merged = combine(merged, node)
            # merged[3] is the maximum subarray sum in the filtered array.
            ans = max(ans, merged[3])
        
        return ans