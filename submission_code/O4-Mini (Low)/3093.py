from typing import List

class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        total = 0
        for i, val in enumerate(nums):
            if bin(i).count('1') == k:
                total += val
        return total

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1
    nums1 = [5, 10, 1, 5, 2]
    k1 = 1
    print(sol.sumIndicesWithKSetBits(nums1, k1))  # Output: 13

    # Example 2
    nums2 = [4, 3, 2, 1]
    k2 = 2
    print(sol.sumIndicesWithKSetBits(nums2, k2))  # Output: 1