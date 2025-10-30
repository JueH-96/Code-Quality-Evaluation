class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        
        # Apply each query in sequence
        for k, (l, r, val) in enumerate(queries):
            # Apply the decrement to each index in the range [l, r]
            for i in range(l, r + 1):
                nums[i] -= val
            
            # Check if the array has become a Zero Array
            if all(x == 0 for x in nums):
                return k + 1  # Return the 1-based index of the query that achieves this
        
        # If no such k is found that makes the array zero, return -1
        return -1