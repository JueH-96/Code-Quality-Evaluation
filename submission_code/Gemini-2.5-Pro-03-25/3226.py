import heapq
from typing import List

class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        """
        Simulates the game between Alice and Bob based on the rules provided.

        Args:
            nums: A 0-indexed integer array of even length.

        Returns:
            The resulting array arr after the game finishes.
        """

        # Sort the array first. This makes finding the minimum element in each step efficient (O(1) access).
        # Sorting takes O(N log N) time.
        nums.sort()

        # Initialize the result array.
        arr = []
        
        # The game continues until nums is empty. Since we process two elements per round,
        # we can iterate through the sorted list taking elements two at a time.
        # The length of nums is guaranteed to be even.
        n = len(nums)
        for i in range(0, n, 2):
            # In each round:
            # 1. Alice removes the minimum element. Since nums is sorted, this is nums[i].
            alice_removed = nums[i]
            
            # 2. Bob removes the *new* minimum element. Since nums is sorted and Alice took nums[i],
            #    the next minimum is nums[i+1].
            bob_removed = nums[i+1]
            
            # 3. Bob appends his removed element to arr first.
            arr.append(bob_removed)
            
            # 4. Alice appends her removed element to arr second.
            arr.append(alice_removed)

        # Return the resulting array arr after all rounds are completed.
        return arr

# Example Usage (for testing outside the class structure if needed):
# sol = Solution()
# nums1 = [5, 4, 2, 3]
# print(f"Input: {nums1}, Output: {sol.numberGame(nums1)}") # Expected: [3, 2, 5, 4]

# nums2 = [2, 5]
# print(f"Input: {nums2}, Output: {sol.numberGame(nums2)}") # Expected: [5, 2]

# nums3 = [1, 2, 3, 4, 5, 6]
# print(f"Input: {nums3}, Output: {sol.numberGame(nums3)}") # Expected: [2, 1, 4, 3, 6, 5]