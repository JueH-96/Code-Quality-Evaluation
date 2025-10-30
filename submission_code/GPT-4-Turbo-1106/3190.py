class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        max_nums1 = max(nums1)
        max_nums2 = max(nums2)
        
        # If the last elements are already the maximums, no operations are needed.
        if nums1[n - 1] == max_nums1 and nums2[n - 1] == max_nums2:
            return 0
        
        # If it's impossible to satisfy the conditions, return -1.
        if nums1[n - 1] > max_nums2 and nums2[n - 1] > max_nums1:
            return -1
        
        # Count the number of operations needed.
        operations = 0
        if nums1[n - 1] != max_nums1:
            # Find the index of the maximum in nums1 and swap it with the last element of nums2.
            index_max_nums1 = nums1.index(max_nums1)
            nums1[index_max_nums1], nums2[index_max_nums1] = nums2[index_max_nums1], nums1[index_max_nums1]
            operations += 1
        
        if nums2[n - 1] != max_nums2:
            # Find the index of the maximum in nums2 and swap it with the last element of nums1.
            index_max_nums2 = nums2.index(max_nums2)
            nums1[index_max_nums2], nums2[index_max_nums2] = nums2[index_max_nums2], nums1[index_max_nums2]
            operations += 1
        
        # After performing the operations, check if the conditions are satisfied.
        if nums1[n - 1] == max(nums1) and nums2[n - 1] == max(nums2):
            return operations
        else:
            return -1