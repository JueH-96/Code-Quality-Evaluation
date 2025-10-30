class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        set_nums1 = set(nums1)
        set_nums2 = set(nums2)
        
        # Elements of nums1 that are in nums2
        common_from_nums1 = sum(1 for x in set_nums1 if x in set_nums2)
        
        # Elements of nums2 that are in nums1
        common_from_nums2 = sum(1 for x in set_nums2 if x in set_nums1)
        
        return [common_from_nums1, common_from_nums2]