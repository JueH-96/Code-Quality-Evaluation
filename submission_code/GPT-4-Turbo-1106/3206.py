class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Convert nums2 to a set for faster lookups
        nums2_set = set(nums2)
        
        # Count the number of elements in nums1 that are in nums2
        count1 = sum(1 for num in nums1 if num in nums2_set)
        
        # Convert nums1 to a set for faster lookups
        nums1_set = set(nums1)
        
        # Count the number of elements in nums2 that are in nums1
        count2 = sum(1 for num in nums2 if num in nums1_set)
        
        # Return the counts as a list
        return [count1, count2]