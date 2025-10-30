class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Convert nums2 to a set for O(1) lookups
        set_nums2 = set(nums2)
        # Count the number of elements in nums1 that are in nums2
        count1 = sum(1 for num in nums1 if num in set_nums2)
        
        # Convert nums1 to a set for O(1) lookups
        set_nums1 = set(nums1)
        # Count the number of elements in nums2 that are in nums1
        count2 = sum(1 for num in nums2 if num in set_nums1)
        
        return [count1, count2]