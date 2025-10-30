class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                nums1 = nums[:i]
                nums2 = nums[i:j]
                nums3 = nums[j:]
                if nums2[:len(nums1)] == nums1 or nums3[:len(nums2)] == nums2:
                    ans += 1
        return ans