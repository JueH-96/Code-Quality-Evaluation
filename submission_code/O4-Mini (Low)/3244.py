from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        # The key is that we can use each occurrence of the minimum value
        # to "absorb" any larger element without reducing the count of mins:
        #   min % larger = min
        # so we pair a min with every non‐min, each such op removes two elements
        # and adds back one min, effectively decreasing total count by 1 but
        # keeping the same number of mins.
        # After eliminating all non‐mins, we are left with only the mins.
        # Now we can pair mins among themselves: min % min = 0,
        # turning each pair of mins into one zero (which cannot be used further).
        # If k = count of mins, we can perform floor(k/2) such ops producing that
        # many zeros, and if k is odd one min remains.  Total survivors = floor(k/2) + (k mod 2) = ceil(k/2).
        #
        # Thus the answer is ceil(freq_min / 2).
        
        if not nums:
            return 0
        mn = min(nums)
        cnt = nums.count(mn)
        # ceil(cnt / 2) = (cnt + 1) // 2 for integers
        return (cnt + 1) // 2