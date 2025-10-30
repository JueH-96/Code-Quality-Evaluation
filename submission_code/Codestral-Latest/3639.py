class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        # Create a difference array to handle range updates efficiently
        n = len(nums)
        diff = [0] * (n + 1)

        # Apply each query to the difference array
        for l, r in queries:
            diff[l] += 1
            if r + 1 < n:
                diff[r + 1] -= 1

        # Reconstruct the array from the difference array
        current = 0
        for i in range(n):
            current += diff[i]
            nums[i] -= current

        # Check if all elements in nums are zero
        return all(x == 0 for x in nums)