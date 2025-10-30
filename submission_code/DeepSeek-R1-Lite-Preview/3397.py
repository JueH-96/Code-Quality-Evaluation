class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Sort both arrays
        sorted_nums1 = sorted(nums1)
        sorted_nums2 = sorted(nums2)
        
        # Calculate x
        x = sorted_nums2[0] - sorted_nums1[0]
        
        # Verify that adding x to each element in sorted_nums1 matches sorted_nums2
        for num1, num2 in zip(sorted_nums1, sorted_nums2):
            if num1 + x != num2:
                # According to the problem constraints, this should not happen
                return -1  # Just in case, though the problem says there is always a valid x
        
        return x