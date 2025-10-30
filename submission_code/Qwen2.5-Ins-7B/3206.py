class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        count1 = sum(any(num in nums2 for num in nums1))
        count2 = sum(any(num in nums1 for num in nums2))
        return [count1, count2]