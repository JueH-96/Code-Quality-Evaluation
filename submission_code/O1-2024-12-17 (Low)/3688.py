class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        """
        We want the maximum subarray sum in nums after optionally removing all
        occurrences of exactly one integer x (or doing no removal at all).
        
        Key idea:
         - The max subarray sum if we do NOT remove anything is a standard
           Kadane's algorithm application.
         - If we remove all occurrences of x, the array is split into segments
           separated by x's positions. The maximum subarray sum in that new
           array is the maximum over all these segments (since any valid subarray
           cannot cross positions where x was removed).
         - To efficiently evaluate this for all possible x, we can build a
           "maximum subarray sum" segment tree for the original array. Then,
           for each x, we identify the segments between occurrences of x
           and query the segment tree for each of these segments. The best
           subarray sum over these segments is what we get if we remove x.
         - We take the maximum over all x of these "remove x" subarray sums
           and also compare with the "no removal" subarray sum. The overall
           maximum is our answer.

        Complexity:
         - Building the segment tree: O(N).
         - We then collect positions of each distinct x. The sum of frequencies
           over all distinct x is N, so we will be querying for O(N) segments total.
         - Each query takes O(log N), giving O(N log N) worst-case time
           which is acceptable for N up to 1e5 in optimized Python (or typical C++).
        """

        # -------------------------------
        # 1) Quick Kadane's for "no removal" case
        # -------------------------------
        best_no_removal = float('-inf')
        curr_sum = 0
        for val in nums:
            curr_sum = max(val, curr_sum + val)
            best_no_removal = max(best_no_removal, curr_sum)

        # Edge case: if there's only one element, that's the answer
        if len(nums) == 1:
            return best_no_removal

        n = len(nums)

        # -------------------------------
        # 2) Build structures for segment-tree-based "range maximum subarray sum"
        #    Each node will store:
        #       total_sum : sum of the entire segment
        #       pref_max  : maximum prefix sum in the segment
        #       suff_max  : maximum suffix sum in the segment
        #       best_max  : maximum subarray sum in the segment
        # -------------------------------
        import math
        size = 1
        while size < n:
            size <<= 1
        
        # We'll store the tree in arrays, 1-indexed for convenience
        # Each node: (total_sum, pref_max, suff_max, best_max)
        INF_NEG = float('-inf')
        segtree = [(0, INF_NEG, INF_NEG, INF_NEG)] * (2 * size)

        # Function to merge two segment nodes
        def merge(left_node, right_node):
            total_sum = left_node[0] + right_node[0]
            pref_max = max(left_node[1], left_node[0] + right_node[1])
            suff_max = max(right_node[2], right_node[0] + left_node[2])
            best_max = max(left_node[3], right_node[3], left_node[2] + right_node[1])
            return (total_sum, pref_max, suff_max, best_max)

        # Build the tree: O(N)
        def build():
            # Initialize the leaves in the second half
            for i in range(n):
                val = nums[i]
                segtree[size + i] = (val, val, val, val)
            for i in range(n, size):
                # For out-of-range positions, treat as "very negative" so they won't affect queries
                segtree[size + i] = (0, INF_NEG, INF_NEG, INF_NEG)
            # Build parents
            for i in range(size - 1, 0, -1):
                segtree[i] = merge(segtree[i << 1], segtree[i << 1 | 1])

        # Query the segment tree for the max subarray sum in [l, r] (0-based, inclusive)
        def query(l, r):
            if l > r:
                return (0, INF_NEG, INF_NEG, INF_NEG)  # "invalid" node
            l += size
            r += size
            left_res = (0, INF_NEG, INF_NEG, INF_NEG)
            right_res = (0, INF_NEG, INF_NEG, INF_NEG)
            while l <= r:
                if (l & 1) == 1:
                    left_res = merge(left_res, segtree[l])
                    l += 1
                if (r & 1) == 0:
                    right_res = merge(segtree[r], right_res)
                    r -= 1
                l >>= 1
                r >>= 1
            res = merge(left_res, right_res)
            return res

        build()

        # -------------------------------
        # 3) Gather positions of each distinct value
        # -------------------------------
        from collections import defaultdict
        positions = defaultdict(list)
        for i, val in enumerate(nums):
            positions[val].append(i)

        # -------------------------------
        # 4) For each distinct x, find the split segments [start,end]
        #    that lie between consecutive occurrences of x, and take
        #    the maximum subarray sum from those segments.
        # -------------------------------
        best_after_removal = float('-inf')
        for x, idxs in positions.items():
            # If x occurs in positions idxs (sorted already by insertion),
            # the segments that remain after removing x are:
            #    [0, idxs[0]-1], [idxs[0]+1, idxs[1]-1], ..., [idxs[-1]+1, n-1]
            # We'll query all those intervals from the segment tree
            prev = -1
            local_best = float('-inf')
            for pos in idxs:
                # segment is [prev+1, pos-1]
                left = prev + 1
                right = pos - 1
                if left <= right:
                    node = query(left, right)
                    local_best = max(local_best, node[3])  # node[3] is best_max
                prev = pos
            # Last segment [prev+1, n-1]
            if prev < n - 1:
                node = query(prev + 1, n - 1)
                local_best = max(local_best, node[3])
            # If x appears throughout the array in a way that leaves no valid segment,
            # local_best might still be -inf. However, the problem states "nums remains
            # non-empty on removing x", so there's at least some segment left. We track
            # that local_best anyway (it can't remain -inf if there's at least one valid segment).
            best_after_removal = max(best_after_removal, local_best)

        # -------------------------------
        # 5) Final result: max of "no removal" and "best removal"
        # -------------------------------
        return max(best_no_removal, best_after_removal)