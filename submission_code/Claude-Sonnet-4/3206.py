class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Convert arrays to sets for O(1) lookup
        set1 = set(nums1)
        set2 = set(nums2)
        
        # Count indices in nums1 whose values appear in nums2
        count1 = sum(1 for num in nums1 if num in set2)
        
        # Count indices in nums2 whose values appear in nums1
        count2 = sum(1 for num in nums2 if num in set1)
        
        return [count1, count2]