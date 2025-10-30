from typing import List
import sys

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        # Edge case: if there's only one element, removing it is not allowed (would be empty),
        # so just return that element.
        n = len(nums)
        if n == 1:
            return nums[0]
        
        #-------------------------------------------------------------------------
        # Build a Segment Tree that can return:
        #   (total_sum, prefix_sum, suffix_sum, best_subarray_sum) for any range
        # so we can quickly get these 4-tuples and merge them.
        #-------------------------------------------------------------------------

        # A large negative constant to represent -∞ in our 4-tuples
        NEG_INF = -10**15
        
        # A helper to make a 4-tuple from a single value
        def make_data(val: int):
            return (val, val, val, val)  # (sum, prefix, suffix, best)
        
        # Identity segment representing an "empty" range
        # sum=0, prefix=-∞, suffix=-∞, best=-∞ ensures it won't affect merges incorrectly
        IDENTITY = (0, NEG_INF, NEG_INF, NEG_INF)
        
        # How to combine two 4-tuples a and b (each = (sum, pref, suff, best))
        def combine(a, b):
            total_sum = a[0] + b[0]
            prefix = max(a[1], a[0] + b[1])
            suffix = max(b[2], b[0] + a[2])
            best_sub = max(a[3], b[3], a[2] + b[1])
            return (total_sum, prefix, suffix, best_sub)
        
        # Build the iterative segment tree
        size = 1
        while size < n:
            size *= 2
        
        segtree = [IDENTITY] * (2 * size)
        
        # Initialize leaves
        for i in range(n):
            segtree[size + i] = make_data(nums[i])
        
        # Build internal nodes
        for i in range(size - 1, 0, -1):
            segtree[i] = combine(segtree[2*i], segtree[2*i + 1])
        
        # Query function for range [l..r], 0-based
        def query(l, r):
            if l > r:
                return IDENTITY
            l += size
            r += size
            left_res = IDENTITY
            right_res = IDENTITY
            while l <= r:
                if (l & 1) == 1:
                    left_res = combine(left_res, segtree[l])
                    l += 1
                if (r & 1) == 0:
                    right_res = combine(segtree[r], right_res)
                    r -= 1
                l >>= 1
                r >>= 1
            return combine(left_res, right_res)
        
        # 1) Maximum subarray sum without any removal
        # The best subarray sum for the entire array is segtree[1][3] if we built a full tree
        best_no_removal = segtree[1][3]
        
        # 2) For each distinct value x, remove all occurrences of x and compute max subarray sum
        from collections import defaultdict
        pos_map = defaultdict(list)
        for i, val in enumerate(nums):
            pos_map[val].append(i)
        
        best_with_removal = best_no_removal
        
        for x, positions in pos_map.items():
            # If removing x empties the array, skip it
            if len(positions) == n:
                continue
            
            # Build intervals that remain after removing all x
            intervals = []
            start = 0
            for p in positions:
                end = p - 1
                if start <= end:
                    intervals.append((start, end))
                start = p + 1
            if start <= n - 1:
                intervals.append((start, n - 1))
            
            # Gather segment info for each interval
            segs = []
            for (l, r) in intervals:
                segs.append(query(l, r))
            
            # Merge all intervals' 4-tuples in a linear pass
            merged = IDENTITY
            for sg in segs:
                merged = combine(merged, sg)
            
            # merged[3] = best subarray sum for the array after removing x
            if merged[3] > best_with_removal:
                best_with_removal = merged[3]
        
        return best_with_removal