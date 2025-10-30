from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        # Compute the sum and count the number of zeros for each array.
        s1 = sum(nums1)
        s2 = sum(nums2)
        count1 = nums1.count(0)
        count2 = nums2.count(0)
        
        # Case 1: Both arrays have no zeros.
        # Then the sums are fixed. They must be equal; otherwise, it is impossible.
        if count1 == 0 and count2 == 0:
            return s1 if s1 == s2 else -1
        
        # Case 2: One array is fixed (no zeros) and the other is adjustable.
        # For an array with zeros, replacing each zero must add at least 1.
        # So the minimal possible sum for an array is its original sum plus the count of zeros.
        if count1 == 0:
            # nums1 is fixed. nums2 can be adjusted.
            # We need the final sum S to be s1.
            # The smallest sum possible for nums2 is s2 + count2.
            # It is possible only if s1 >= s2 + count2.
            if s1 >= s2 + count2:
                return s1
            else:
                return -1
        
        if count2 == 0:
            # nums2 is fixed. nums1 can be adjusted.
            if s2 >= s1 + count1:
                return s2
            else:
                return -1
        
        # Case 3: Both arrays have one or more zeros.
        # Each array can have its zeros replaced by any strictly positive integers.
        # Thus, the sum for nums1 can be any integer >= (s1 + count1)
        # and the sum for nums2 can be any integer >= (s2 + count2).
        # To have a common sum, we must choose S such that:
        #   S >= s1 + count1 and S >= s2 + count2.
        # The minimal equal S is therefore:
        return max(s1 + count1, s2 + count2)

# For quick testing:
if __name__ == "__main__":
    sol = Solution()
    # Example 1: Expected output 12.
    print(sol.minSum([3,2,0,1,0], [6,5,0]))  # Output => 12
    
    # Example 2: Expected output -1.
    print(sol.minSum([2,0,2,0], [1,4]))        # Output => -1