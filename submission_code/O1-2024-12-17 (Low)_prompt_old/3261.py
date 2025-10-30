class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        """
        We want to merge up to k pairs of adjacent elements (each merge being an AND)
        so that the bitwise-OR of the resulting array is as small as possible.
        
        Observing that the result can never exceed the OR of all elements (since merges only
        clear bits) and can never be lower than the global AND of all elements (since with
        at most k < n merges, we can reduce the length at most to 1, whose value is the AND
        of some subset of elements), we can do a binary-search (or step through possible values)
        for the smallest "target" T such that it is possible to form a final array (with at
        most k merges) where every element is at most T. If that is possible, we decrease T;
        if not, we increase T. The minimal T for which it is possible is our answer.
        
        To test feasibility of a candidate 'mid' (i.e. "can we ensure all final elements ≤ mid"?),
        we perform a left-to-right greedy simulation:
        
        - We iterate over nums, building "final elements". Let cur = nums[i].
        - We try NOT to merge if cur|nums[i+1] ≤ mid (i.e. both can remain separate final elements).
        - If cur|nums[i+1] > mid, we must merge cur with nums[i+1] (cur = cur & nums[i+1]) to try
          to clear bits. That uses up one merge. We keep merging with the next adjacent element
          so long as cur is still > mid (and we have merges left). If at any point merges exceed k,
          or we cannot get cur ≤ mid with merges on consecutive elements, we fail for this 'mid'.
        - If we finish scanning without using more than k merges, it's feasible.
        
        We'll do a straightforward binary search over [0.. max_possible], where max_possible
        is the OR of all input elements. The answer is the smallest feasible mid.
        """
        
        import math
        
        # Edge cases
        if len(nums) == 1:
            # Only one element, no merges matter
            return nums[0]
        if k == 0:
            # No merges allowed; answer is just OR of all elements
            ans = 0
            for x in nums:
                ans |= x
            return ans
        
        max_val = 0
        for x in nums:
            max_val |= x

        # A helper function to check feasibility for a given candidate 'mid'.
        def canDo(mid: int) -> bool:
            merges_used = 0
            i = 0
            n = len(nums)
            while i < n:
                cur = nums[i]
                # Move to next element(s) to see if we need merges
                j = i + 1
                while j < n:
                    # If OR is within mid, we do NOT merge these elements
                    # -> we finalize 'cur' as one final element and break
                    # so that nums[j] can start a new final element on its own.
                    if (cur | nums[j]) <= mid:
                        break
                    # Otherwise we must merge to try clearing bits
                    merges_used += 1
                    cur = cur & nums[j]
                    if merges_used > k:
                        return False
                    j += 1
                    # If after merging we are <= mid, we can consider this final
                    if cur <= mid:
                        break
                # If after merging (or not merging) 'cur' is still > mid, fail
                if cur > mid:
                    return False
                # Move i to j (since we've decided 'cur' is our final element and
                # anything up to j-1 was merged into 'cur')
                i = max(j, i+1)
            return True
        
        # Binary search [0..max_val]
        lo, hi = 0, max_val
        ans = max_val
        while lo <= hi:
            mid = (lo + hi) // 2
            if canDo(mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans