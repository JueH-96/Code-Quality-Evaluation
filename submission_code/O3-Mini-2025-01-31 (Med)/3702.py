from math import gcd
from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        n = len(nums)
        max_len = 0
        # Iterate over every possible subarray starting index
        for i in range(n):
            prod = 1
            current_gcd = 0  # We'll define gcd(0, a) = a
            current_lcm = 1
            for j in range(i, n):
                a = nums[j]
                prod *= a
                current_gcd = a if current_gcd == 0 else gcd(current_gcd, a)
                # To update lcm: lcm(x, y) = x * y // gcd(x, y)
                # For the first element, current_lcm is 1; so lcm(1, a) = a.
                current_lcm = (current_lcm * a) // gcd(current_lcm, a)
                
                # Check condition: prod == current_gcd * current_lcm
                if prod == current_gcd * current_lcm:
                    max_len = max(max_len, j - i + 1)
        return max_len

# Below are some tests to verify the solution.
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1:
    nums1 = [1,2,1,2,1,1,1]
    print(sol.maxLength(nums1))  # Expected output: 5

    # Example 2:
    nums2 = [2,3,4,5,6]
    print(sol.maxLength(nums2))  # Expected output: 3

    # Example 3:
    nums3 = [1,2,3,1,4,5,1]
    print(sol.maxLength(nums3))  # Expected output: 5

    # Additional tests:
    # Single element subarrays only product equivalent if the element is 1
    nums4 = [2, 1, 3, 1, 1]
    # For nums4, valid subarrays at single positions: index 1 (1), index 3 (1), index 4 (1)
    # Also, any subarray of length 2 is always product equivalent.
    print(sol.maxLength(nums4))  # Expected output should be at least 2 because of all pairs.