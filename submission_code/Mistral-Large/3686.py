from typing import List

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        beautiful_count = 0

        # Generate all possible prefixes
        prefixes = []
        for i in range(n):
            prefixes.append(tuple(nums[:i+1]))

        # Check all possible splits
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                nums1 = tuple(nums[:i])
                nums2 = tuple(nums[i:j])
                nums3 = tuple(nums[j:])

                if nums1 in prefixes and (nums1 == nums2[:len(nums1)] or nums2 == nums3[:len(nums2)]):
                    beautiful_count += 1

        return beautiful_count