class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        S1 = sum(x for x in nums1 if x != 0)
        S2 = sum(x for x in nums2 if x != 0)
        z1 = sum(1 for x in nums1 if x == 0)
        z2 = sum(1 for x in nums2 if x == 0)
        
        # Case 1: No zeros in either array => must already be equal
        if z1 == 0 and z2 == 0:
            return S1 if S1 == S2 else -1
        
        # Case 2: nums1 fixed, nums2 can grow
        if z1 == 0 and z2 > 0:
            # T must be S1, so check if S1 >= S2 + z2
            if S1 >= S2 + z2:
                return S1
            else:
                return -1
        
        # Case 3: nums2 fixed, nums1 can grow
        if z2 == 0 and z1 > 0:
            # T must be S2, so check if S2 >= S1 + z1
            if S2 >= S1 + z1:
                return S2
            else:
                return -1
        
        # Case 4: Both can grow
        # T = max(S1 + z1, S2 + z2)
        return max(S1 + z1, S2 + z2)