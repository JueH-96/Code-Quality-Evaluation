class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        set_nums2 = set(nums2)
        count_in_nums2 = sum(1 for x in nums1 if x in set_nums2)
        
        set_nums1 = set(nums1)
        count_in_nums1 = sum(1 for x in nums2 if x in set_nums1)
        
        return [count_in_nums2, count_in_nums1]