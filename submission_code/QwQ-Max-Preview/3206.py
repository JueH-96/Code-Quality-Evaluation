class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Create a set of elements in nums2 for quick lookup
        set_nums2 = set(nums2)
        # Count elements in nums1 that are present in nums2
        count1 = sum(1 for num in nums1 if num in set_nums2)
        
        # Create a set of elements in nums1 for quick lookup
        set_nums1 = set(nums1)
        # Count elements in nums2 that are present in nums1
        count2 = sum(1 for num in nums2 if num in set_nums1)
        
        return [count1, count2]