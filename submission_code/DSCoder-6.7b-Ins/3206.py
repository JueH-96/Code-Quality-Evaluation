class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Create a set from nums1 and nums2
        set1 = set(nums1)
        set2 = set(nums2)
        
        # Calculate the number of indices i such that 0 <= i < n and nums1[i] occurs at least once in nums2
        count1 = len([i for i in range(len(nums1)) if nums1[i] in set2])
        
        # Calculate the number of indices i such that 0 <= i < m and nums2[i] occurs at least once in nums1
        count2 = len([i for i in range(len(nums2)) if nums2[i] in set1])
        
        # Return the counts as a list
        return [count1, count2]