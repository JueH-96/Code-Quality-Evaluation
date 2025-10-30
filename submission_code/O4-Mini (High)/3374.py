from typing import List

class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        i = 0
        # Split nums into maximal alternating segments
        while i < n:
            j = i
            # extend as long as the next element is different
            while j + 1 < n and nums[j] != nums[j + 1]:
                j += 1
            length = j - i + 1
            # number of subarrays in a segment of length L = L*(L+1)//2
            ans += length * (length + 1) // 2
            i = j + 1
        return ans