class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        """
        We keep track of which indices are peaks using a Fenwick (Binary Indexed) Tree
        so that we can quickly (O(log n)) update and query the total count of peaks
        in any subarray. A "peak" index i satisfies:
            0 < i < len(nums) - 1 and nums[i] > nums[i-1] and nums[i] > nums[i+1].
        For a subarray [l..r], only indices in (l, r) can be peaks in that subarray,
        since the ends can't be counted as peaks.
        """

        n = len(nums)
        
        # Helper function to determine if i is a peak.
        def is_peak(i):
            if i <= 0 or i >= n-1:
                return False
            return nums[i] > nums[i-1] and nums[i] > nums[i+1]
        
        # Fenwick Tree (Binary Indexed Tree) for peak counts.
        fenwicks = [0]*(n+1)
        
        # Fenwick Tree functions:
        def fenwicks_update(i, diff):
            # i is 0-based; Fenwicks uses 1-based indexing
            i += 1
            while i <= n:
                fenwicks[i] += diff
                i += i & -i
        
        def fenwicks_prefix_sum(i):
            # sum up to and including i (0-based index in peak array)
            # in Fenwicks we go up to i+1
            i += 1
            result = 0
            while i > 0:
                result += fenwicks[i]
                i -= i & -i
            return result
        
        def fenwicks_range_sum(left, right):
            if left > right:
                return 0
            return fenwicks_prefix_sum(right) - fenwicks_prefix_sum(left-1)
        
        # Initialize the peak array and Fenwicks tree
        peak = [0]*n
        for i in range(1, n-1):
            if is_peak(i):
                peak[i] = 1
        
        # Build Fenwicks from peak array
        for i in range(n):
            if peak[i] == 1:
                fenwicks_update(i, 1)
        
        answers = []
        
        # Process queries
        for t, x, y in queries:
            if t == 1:
                # Query type 1: count peaks in subarray nums[x..y]
                # The subarray's potential peaks are in [x+1..y-1].
                if (y - x) < 2:
                    # If subarray length < 3, no possible peaks
                    answers.append(0)
                else:
                    # Count how many peaks in [x+1..y-1]
                    left = x + 1
                    right = y - 1
                    ans = fenwicks_range_sum(left, right)
                    answers.append(ans)
            
            else:
                # Query type 2: update nums[x] = y
                idx, val = x, y
                nums[idx] = val
                # Re-check up to 3 positions that can change peak status
                for i_to_check in [idx-1, idx, idx+1]:
                    if 1 <= i_to_check <= n-2:
                        old_val = peak[i_to_check]
                        new_val = 1 if is_peak(i_to_check) else 0
                        if old_val != new_val:
                            peak[i_to_check] = new_val
                            fenwicks_update(i_to_check, new_val - old_val)
        
        return answers