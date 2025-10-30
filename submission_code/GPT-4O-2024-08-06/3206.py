class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Convert nums2 to a set for fast lookup
        set_nums2 = set(nums2)
        
        # Count how many elements in nums1 are present in nums2
        count1 = sum(1 for num in nums1 if num in set_nums2)
        
        # Convert nums1 to a set for fast lookup
        set_nums1 = set(nums1)
        
        # Count how many elements in nums2 are present in nums1
        count2 = sum(1 for num in nums2 if num in set_nums1)
        
        # Return the result as a list
        return [count1, count2]