class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Sort both arrays to compare them in a defined order
        s1, s2 = sorted(nums1), sorted(nums2)
        
        # The integer x is the difference between the first elements
        x = s2[0] - s1[0]
        
        # Verify that adding x to s1 indeed produces s2
        for a, b in zip(s1, s2):
            if a + x != b:
                # Given the problem guarantees a valid x, 
                # this situation should never occur. 
                # But if it does, we might raise an error or return something indicative.
                raise ValueError("No valid solution found, though one was guaranteed.")
        
        return x