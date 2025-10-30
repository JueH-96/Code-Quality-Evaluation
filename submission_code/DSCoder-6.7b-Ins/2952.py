class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        nums = sorted([(a, b) for a, b in zip(nums1, nums2)], reverse=True)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i][1]
        for i in range(n):
            if nums[i][0] + nums[i][1] * (i + 1) + prefix[i] <= x:
                return i
        return -1