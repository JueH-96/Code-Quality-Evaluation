class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        max1, max2 = max(nums1), max(nums2)
        
        # Check if [max1, max2] or [max2, max1] is the desired result
        if nums1[n - 1] == max1 and nums2[n - 1] == max2:
            return 0
        elif nums1[n - 1] == max2 and nums2[n - 1] == max1:
            return 1
        
        # Check if the desired results are achievable
        if max1 not in nums1 or max2 not in nums2:
            return -1
        
        # Count operations needed for [max1, max2]
        operations1 = 0
        for i in range(n):
            if nums1[i] > max1 or nums2[i] > max2:
                return -1
            if nums1[i] > max1 and nums2[i] > max2:
                return -1
            if i == n - 1 or nums1[i] >= max2 or nums2[i] >= max1:
                operations1 += 1
        
        # Count operations needed for [max2, max1] and choose the minimum
        operations2 = 0
        for i in range(n):
            if nums1[i] > max2 or nums2[i] > max1:
                return -1
            if nums1[i] > max1 and nums2[i] > max2:
                return -1
            if i == n - 1 or nums1[i] >= max1 or nums2[i] >= max2:
                operations2 += 1
        
        return min(operations1, operations2)