class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        nums1 = [-1 * x for x in nums1]
        nums2 = [-1 * x for x in nums2]
        s1 = set(nums1)
        s2 = set(nums2)
        s = s1.union(s2)
        return len(s)