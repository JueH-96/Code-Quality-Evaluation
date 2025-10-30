from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        # Compute sums and zero counts for each array
        s1 = sum(x for x in nums1 if x != 0)
        z1 = nums1.count(0)
        s2 = sum(x for x in nums2 if x != 0)
        z2 = nums2.count(0)

        # Case 1: no zeros in either array
        if z1 == 0 and z2 == 0:
            return s1 if s1 == s2 else -1

        # Case 2: zeros only in nums2
        if z1 == 0:
            # We must have total sum T = s1
            # And we need s1 >= s2 + z2 so that sum of z2 positive ints can reach (s1 - s2)
            if s1 < s2 + z2:
                return -1
            return s1

        # Case 3: zeros only in nums1
        if z2 == 0:
            # T = s2
            if s2 < s1 + z1:
                return -1
            return s2

        # Case 4: both have at least one zero
        # We can choose T as small as max(s1+z1, s2+z2)
        return max(s1 + z1, s2 + z2)