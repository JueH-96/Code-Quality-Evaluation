from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Create a set of required elements: {1, 2, ..., k}
        required = set(range(1, k + 1))
        operations = 0
        
        # Traverse the array from the end (simulating removal of last element)
        for num in reversed(nums):
            operations += 1
            # If this number is required, remove it from the set.
            if num in required:
                required.remove(num)
            # If we have collected all required elements, break
            if not required:
                break
        
        return operations

# Example usage:
if __name__ == "__main__":
    sol = Solution()

    # Example 1
    nums1 = [3, 1, 5, 4, 2]
    k1 = 2
    print(sol.minOperations(nums1, k1))  # Expected output: 4

    # Example 2
    nums2 = [3, 1, 5, 4, 2]
    k2 = 5
    print(sol.minOperations(nums2, k2))  # Expected output: 5

    # Example 3
    nums3 = [3, 2, 5, 3, 1]
    k3 = 3
    print(sol.minOperations(nums3, k3))  # Expected output: 4