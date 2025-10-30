class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Convert both arrays to sets for efficient membership checking
        set_nums1 = set(nums1)
        set_nums2 = set(nums2)
        
        # Count indices in nums1 where element exists in nums2
        count1 = sum(1 for num in nums1 if num in set_nums2)
        
        # Count indices in nums2 where element exists in nums1
        count2 = sum(1 for num in nums2 if num in set_nums1)
        
        return [count1, count2]