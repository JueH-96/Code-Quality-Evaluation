from typing import List

class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        """
        Dynamic programming.
        dp1 – length of the longest non-decreasing subarray that ends at the current index
               if we take nums1[i] as the value of nums3[i]
        dp2 – the same, but we take nums2[i].
        Only the previous pair (dp1_prev, dp2_prev) is required, so we store two scalars.
        """
        n = len(nums1)
        # length is at least 1 because arrays are non-empty
        dp1 = dp2 = ans = 1                     # values for index 0
        for i in range(1, n):
            new1 = 1                            # choose nums1[i]
            if nums1[i] >= nums1[i - 1]:
                new1 = max(new1, dp1 + 1)
            if nums1[i] >= nums2[i - 1]:
                new1 = max(new1, dp2 + 1)

            new2 = 1                            # choose nums2[i]
            if nums2[i] >= nums1[i - 1]:
                new2 = max(new2, dp1 + 1)
            if nums2[i] >= nums2[i - 1]:
                new2 = max(new2, dp2 + 1)

            ans = max(ans, new1, new2)          # best subarray seen so far
            dp1, dp2 = new1, new2               # slide window

        return ans