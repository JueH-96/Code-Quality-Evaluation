class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Sort both arrays
        sorted_nums1 = sorted(nums1)
        sorted_nums2 = sorted(nums2)
        
        # Calculate x based on the first elements
        x = sorted_nums2[0] - sorted_nums1[0]
        
        # Verify that adding x to each element in nums1 matches nums2
        for num1, num2 in zip(sorted_nums1, sorted_nums2):
            if num1 + x != num2:
                # According to the problem, this should never happen
                raise ValueError("No such integer x exists.")
        
        return x