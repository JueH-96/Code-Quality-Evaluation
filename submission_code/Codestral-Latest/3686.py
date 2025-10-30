from typing import List

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0

        # Calculate the prefix sums
        prefix_sums = [0] * (n + 1)
        for i in range(n):
            prefix_sums[i + 1] = prefix_sums[i] + nums[i]

        # Check all possible splits
        for i in range(1, n):
            for j in range(i + 1, n):
                nums1 = prefix_sums[i] - prefix_sums[0]
                nums2 = prefix_sums[j] - prefix_sums[i]
                nums3 = prefix_sums[n] - prefix_sums[j]

                if nums1 == nums2 or nums2 == nums3:
                    count += 1

        return count