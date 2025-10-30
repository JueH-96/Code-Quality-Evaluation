class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        count = 0
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                nums1 = nums[:i]
                nums2 = nums[i:j]
                nums3 = nums[j:]
                if not nums1 or not nums2 or not nums3:
                    continue
                is_nums1_prefix_nums2 = False
                if len(nums1) <= len(nums2):
                    is_nums1_prefix_nums2 = True
                    for k in range(len(nums1)):
                        if nums1[k] != nums2[k]:
                            is_nums1_prefix_nums2 = False
                            break
                is_nums2_prefix_nums3 = False
                if len(nums2) <= len(nums3):
                    is_nums2_prefix_nums3 = True
                    for k in range(len(nums2)):
                        if nums2[k] != nums3[k]:
                            is_nums2_prefix_nums3 = False
                            break
                if is_nums1_prefix_nums2 or is_nums2_prefix_nums3:
                    count += 1
        return count