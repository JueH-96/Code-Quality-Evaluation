class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        # Create a difference array to efficiently handle range updates
        diff = [0] * (len(nums) + 1)
        
        # Apply the queries to the difference array
        for l, r in queries:
            diff[l] -= 1
            if r + 1 < len(nums):
                diff[r + 1] += 1
        
        # Apply the difference array to the original array
        for i in range(len(nums)):
            if i > 0:
                diff[i] += diff[i - 1]
            nums[i] += diff[i]
        
        # Check if the resulting array is a zero array
        return all(num == 0 for num in nums)