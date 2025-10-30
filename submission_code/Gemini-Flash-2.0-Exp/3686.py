class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                nums1 = nums[:i]
                nums2 = nums[i:j]
                nums3 = nums[j:]

                if len(nums1) <= len(nums2):
                    if nums1 == nums2[:len(nums1)]:
                        count += 1
                        continue
                if len(nums2) <= len(nums3):
                    if nums2 == nums3[:len(nums2)]:
                        count += 1
                        continue
        return count