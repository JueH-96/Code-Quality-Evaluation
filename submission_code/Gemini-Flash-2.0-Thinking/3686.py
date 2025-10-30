class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(1, n):  # End of nums1
            for j in range(i + 1, n):  # End of nums2
                len1 = i
                len2 = j - i
                len3 = n - j

                nums1 = nums[:i]
                nums2 = nums[i:j]
                nums3 = nums[j:]

                # Check if nums1 is a prefix of nums2
                is_prefix1 = False
                if len(nums1) <= len(nums2) and nums2[:len(nums1)] == nums1:
                    is_prefix1 = True

                # Check if nums2 is a prefix of nums3
                is_prefix2 = False
                if len(nums2) <= len(nums3) and nums3[:len(nums2)] == nums2:
                    is_prefix2 = True

                if is_prefix1 or is_prefix2:
                    count += 1
        return count