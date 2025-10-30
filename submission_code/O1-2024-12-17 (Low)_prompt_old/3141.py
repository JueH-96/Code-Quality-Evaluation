class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        """
        We have an infinite repetition of nums, and we want the shortest contiguous subarray
        whose sum is exactly target. If not possible, return -1.
        
        Key ideas:
          1) Let s = sum(nums). Then target can be decomposed as target = q*s + r, where
             q = target // s, r = target % s.
          2) A contiguous subarray that covers q full repeats of nums has length q*n if
             we must take all elements in those repeats (since skipping elements in the
             middle would break contiguousness). Past those q full repeats, we may only
             need some extra part of the (q+1)-th repetition to make up remainder r.
          3) However, it might be beneficial to consider (q+1) repeats if r is large,
             or even (q-1) repeats plus a subarray summing to (s+r), etc.  But to keep
             the solution simple and still correct:
                - We define a set of possible multiples K = {q, q+1} (if q > 0), or {0,1}
                  if q=0.
                - For each k in K, leftover = target - k*s. If leftover < 0, skip.
                  If leftover = 0, we can form the sum by exactly k full arrays,
                  so candidate length = k*n (if k>0). 
                  Otherwise, if leftover>0, we try to find a sub-subarray of nums+nums
                  that sums to leftover.
                - We pick the minimal candidate across all feasible k.
        
        The sub-subarray sum to leftover is found via a standard "sliding window" 
        (two-pointer) approach over nums extended twice (nums+nums). We only need 2 copies
        because any subarray that "wraps around" can be captured in at most 2 concatenations.
        
        Overall complexity: O(n) for the sliding window. n can be up to 1e5, which is feasible.
        """
        import sys
        INF = 10**15
        
        s = sum(nums)
        n = len(nums)
        
        # If sum of nums >= target, we still do the same logic of q = target // s etc.
        q, r = divmod(target, s)
        
        # Build the extended array (at most 2n)
        extended = nums + nums
        
        # A helper function to find smallest subarray that sums to 'needed' in 'extended'
        # using a sliding window. Returns INF if not found.
        def min_subarray_sum(needed: int) -> int:
            if needed == 0:
                # We can't have a subarray of positive length summing to 0
                # given all nums[i] >= 1, so no sub-subarray can have sum=0.
                return INF
            left = 0
            window_sum = 0
            min_len = INF
            for right in range(len(extended)):
                window_sum += extended[right]
                while window_sum > needed and left <= right:
                    window_sum -= extended[left]
                    left += 1
                if window_sum == needed:
                    min_len = min(min_len, right - left + 1)
            return min_len
        
        # We'll try k in {q, q+1} if q>0, or {0,1} if q=0.
        if q == 0:
            candidates = [0, 1]
        else:
            candidates = [q, q+1]
        
        ans = INF
        
        for k in candidates:
            leftover = target - k*s
            if leftover < 0:
                # Not feasible if leftover < 0
                continue
            if leftover == 0:
                # If leftover=0 and k>0, we can achieve target with k full arrays
                # => length = k*n
                if k > 0:
                    ans = min(ans, k*n)
            else:
                # leftover > 0, find smallest sub-subarray in extended
                length_sub = min_subarray_sum(leftover)
                if length_sub != INF:
                    # total length is k full arrays plus that leftover sub-subarray
                    candidate_length = k*n + length_sub
                    ans = min(ans, candidate_length)
        
        return ans if ans != INF else -1