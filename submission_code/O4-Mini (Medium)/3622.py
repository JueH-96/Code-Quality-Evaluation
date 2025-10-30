from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        # We build intervals [nums[i]-k, nums[i]+k] for each element.
        # We then sweep-line count how many intervals cover each possible value x.
        # For each x, let cover[x] = number of elements whose interval covers x,
        # and counts[x] = number of elements exactly equal to x.
        # We can change at most numOperations elements (distinct indices) to x,
        # so the max frequency of x we can achieve is
        #   min( cover[x], counts[x] + numOperations )
        # because counts[x] elements already equal x (no op needed), and
        # we can convert up to numOperations other coverable elements.
        # We sweep over all x and take the max.
        
        # Edge case: no operations or empty list
        n = len(nums)
        if n == 0:
            return 0
        # Determine an offset so all interval endpoints are non-negative indices
        offset = k
        max_num = max(nums)
        # The maximum endpoint after offset is (max_num + k + offset)
        size = max_num + k + offset + 3
        
        # Sweep-line array for interval coverings
        cover = [0] * size
        # Direct counts of original values
        counts = [0] * size
        
        # Mark intervals and record counts
        for a in nums:
            l = a - k + offset
            r = a + k + offset
            # l and r are guaranteed >= 0 because offset = k >= nums[i]-k in worst case
            cover[l] += 1
            if r + 1 < size:
                cover[r + 1] -= 1
            counts[a + offset] += 1
        
        # Build prefix sums to get coverage at each x-index
        for i in range(1, size):
            cover[i] += cover[i - 1]
        
        # Compute the answer by checking each possible x (represented by index i)
        ans = 0
        # freq_x = min( cover[i], counts[i] + numOperations )
        for i in range(size):
            c = cover[i]
            if c == 0:
                continue
            # how many already equal to x
            base = counts[i]
            # we can boost up to numOperations of the others
            freq_x = base + min(numOperations, c - base)
            # which is the same as min(c, base + numOperations)
            # freq_x = min(c, base + numOperations)
            if freq_x > ans:
                ans = freq_x
        
        return ans