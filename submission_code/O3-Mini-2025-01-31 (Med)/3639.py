from typing import List

class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        # Create a difference array to efficiently calculate coverage counts.
        diff = [0] * (n + 1)
        
        # For each query, add +1 at l and -1 past r so that prefix summing gives the count.
        for l, r in queries:
            diff[l] += 1
            if r + 1 < n:
                diff[r + 1] -= 1
        
        # Calculate the coverage for each index and check if it meets the requirement.
        # Each index i must appear in at least nums[i] queries (so that we have enough opportunities
        # to decrement it one per query).
        current = 0
        for i in range(n):
            current += diff[i]
            if nums[i] > current:
                return False  # Not enough queries to fully decrement nums[i] to 0.
        
        return True
        
# Example usage:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1:
    nums1 = [1, 0, 1]
    queries1 = [[0, 2]]
    print(sol.isZeroArray(nums1, queries1))  # Expected output: True
    
    # Example 2:
    nums2 = [4, 3, 2, 1]
    queries2 = [[1, 3], [0, 2]]
    print(sol.isZeroArray(nums2, queries2))  # Expected output: False