class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        count1 = 0
        count2 = 0
        
        # Count the number of elements in nums1 that occur in nums2
        for num in nums1:
            if num in nums2:
                count1 += 1
        
        # Count the number of elements in nums2 that occur in nums1
        for num in nums2:
            if num in nums1:
                count2 += 1
        
        return [count1, count2]