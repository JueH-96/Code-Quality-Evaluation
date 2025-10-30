from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)

        # Find the maximum values in nums1 and nums2
        max1 = max(nums1)
        max2 = max(nums2)

        # Check if it's possible to satisfy both conditions
        if nums1[-1] == max1 and nums2[-1] == max2:
            return 0
        if nums1[-1] != max1 and nums2[-1] != max2 and max1 != max2:
            return -1

        # Initialize the number of operations
        operations = 0

        # Swap elements to satisfy the conditions
        for i in range(n):
            if nums1[i] == max1 and nums2[i] == max2:
                nums1[i], nums2[i] = nums2[i], nums1[i]
                operations += 1
                if nums1[-1] == max1 and nums2[-1] == max2:
                    return operations

        return -1

# Test cases
sol = Solution()
print(sol.minOperations([1,2,7], [4,5,3]))  # Output: 1
print(sol.minOperations([2,3,4,5,9], [8,8,4,4,4]))  # Output: 2
print(sol.minOperations([1,5,4], [2,5,3]))  # Output: -1