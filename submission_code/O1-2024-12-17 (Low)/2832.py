class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        """
        We want the longest subarray that can become all one value after removing at most k elements
        from the array. Once we remove some elements, the remaining elements "close-up" and become
        contiguous in the new array.

        Key idea:
          1) Identify all the (zero-based) positions where each distinct value occurs.
          2) If we focus on a particular value v, its positions might look like p = [p0, p1, ..., pm-1].
             A block of these positions p[L]..p[R] in the original array would occupy a segment
             from index p[L] to p[R], which has length segment_len = p[R] - p[L] + 1.
             That segment contains exactly (R - L + 1) occurrences of v. So there are
             gaps = segment_len - (R - L + 1) positions in that segment that are not v,
             and we must remove all of them if we want the final subarray to be entirely v.
          3) We require gaps <= k, i.e. p[R] - p[L] + 1 - (R - L + 1) <= k, so we can afford
             to remove those non-v elements. In the new array (after removals), the elements
             at positions p[L], p[L+1], ..., p[R] become consecutive, forming a subarray
             of length (R - L + 1) all equal to v.
          4) We do a two-pointer sliding window over the sorted positions of each value, tracking
             the largest window [L..R] such that the above condition holds. The maximum (R - L + 1)
             over all values is our answer.

        This runs in O(n) time overall, since each index belongs to exactly one value's position list.
        """
        from collections import defaultdict
        
        # Step 1: gather positions of each distinct value
        positions = defaultdict(list)
        for i, val in enumerate(nums):
            positions[val].append(i)
        
        ans = 0
        
        # Step 2: for each distinct value, use a two-pointer approach
        for val, pos_list in positions.items():
            L = 0
            for R in range(len(pos_list)):
                # While the gap in [pos_list[L], pos_list[R]] is too large, move L
                while (pos_list[R] - pos_list[L] + 1) - (R - L + 1) > k:
                    L += 1
                # The length of this subarray of 'val' is (R - L + 1)
                ans = max(ans, R - L + 1)
        
        return ans