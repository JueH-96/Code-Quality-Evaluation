class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        set_nums2 = set(nums2)
        set_nums1 = set(nums1)
        
        # Count how many elements in nums1 appear in nums2
        count_nums1_in_nums2 = sum(1 for x in nums1 if x in set_nums2)
        # Count how many elements in nums2 appear in nums1
        count_nums2_in_nums1 = sum(1 for x in nums2 if x in set_nums1)
        
        return [count_nums1_in_nums2, count_nums2_in_nums1]