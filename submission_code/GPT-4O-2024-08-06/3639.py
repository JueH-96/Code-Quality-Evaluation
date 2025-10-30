class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        # We will use a difference array to efficiently apply the decrements
        n = len(nums)
        diff = [0] * (n + 1)
        
        # Apply each query to the difference array
        for l, r in queries:
            diff[l] -= 1
            if r + 1 < n:
                diff[r + 1] += 1
        
        # Apply the difference array to the original nums array
        current_decrement = 0
        for i in range(n):
            current_decrement += diff[i]
            nums[i] += current_decrement
        
        # Check if all elements in nums are zero
        return all(x == 0 for x in nums)