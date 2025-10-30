from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        # Compute the sum of non-zero elements and count zeros for both arrays.
        s1 = sum(x for x in nums1 if x != 0)
        z1 = nums1.count(0)
        s2 = sum(x for x in nums2 if x != 0)
        z2 = nums2.count(0)

        # Case 1: Both arrays contain no zeros.
        # Their sums are fixed. We can only equalize if the sums are already equal.
        if z1 == 0 and z2 == 0:
            return s1 if s1 == s2 else -1

        # Case 2: Only one array contains zeros.
        # Then the sum for the array without zeros is fixed.
        # Suppose nums1 is fixed (z1 == 0) and nums2 has zeros.
        # For nums2, even if we replace every zero with 1 (the minimal replacement)
        # the smallest sum achievable is s2 + z2. To equal s1, we require that:
        # s1 >= s2 + z2. If so, we can distribute the extra (s1 - s2) among zeros.
        if z1 == 0:
            return s1 if s1 >= s2 + z2 else -1
        if z2 == 0:
            return s2 if s2 >= s1 + z1 else -1

        # Case 3: Both arrays have zeros.
        # For any array with non-zero sum s and z zeros, the minimal achievable sum is s+z
        # (by replacing each zero with 1). Since we can add any extra to zeros (making one
        # or more replacements arbitrarily larger), the minimum equal sum S that we can obtain 
        # is S = max(s1+z1, s2+z2). This is because one array may already force a higher sum.
        return max(s1 + z1, s2 + z2)

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.minSum([3,2,0,1,0], [6,5,0]))  # Expected output: 12
    # Example 2:
    print(sol.minSum([2,0,2,0], [1,4]))        # Expected output: -1