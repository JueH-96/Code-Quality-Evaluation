class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(1, n):
            for j in range(i + 1, n):
                nums1 = nums[:i]
                nums2 = nums[i:j]
                nums3 = nums[j:]

                if (len(nums1) <= len(nums2) and nums1 == nums2[:len(nums1)]) or \
                   (len(nums2) <= len(nums3) and nums2 == nums3[:len(nums2)]):
                    ans += 1
        return ans