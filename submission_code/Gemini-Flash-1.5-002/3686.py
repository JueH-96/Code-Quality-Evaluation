class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(1, n):
            for j in range(i + 1, n + 1):
                nums1 = nums[:i]
                nums2 = nums[i:j]
                nums3 = nums[j:]

                #Check if nums1 is a prefix of nums2
                is_prefix1 = True
                for k in range(len(nums1)):
                    if k >= len(nums2) or nums1[k] != nums2[k]:
                        is_prefix1 = False
                        break

                #Check if nums2 is a prefix of nums3
                is_prefix2 = True
                for k in range(len(nums2)):
                    if k >= len(nums3) or nums2[k] != nums3[k]:
                        is_prefix2 = False
                        break

                if is_prefix1 or is_prefix2:
                    count += 1
        return count