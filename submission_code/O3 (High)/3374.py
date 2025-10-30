from typing import List

class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        """
        Count subarrays in which no two consecutive elements are equal.
        Because a maximal segment with alternating neighbours contributes
        all its subarrays, we only need the lengths of those segments.
        """
        n = len(nums)
        if n == 0:
            return 0
        
        total = 0          # answer
        seg_len = 1        # current alternating segment length
        
        for i in range(1, n):
            if nums[i] != nums[i - 1]:
                # still alternating, extend current segment
                seg_len += 1
            else:
                # segment ended, add its contribution
                total += seg_len * (seg_len + 1) // 2
                seg_len = 1                       # restart segment
        
        # add the last segment
        total += seg_len * (seg_len + 1) // 2
        return total