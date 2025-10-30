class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        # Iterate over each query
        for k, (l, r, val) in enumerate(queries):
            # Apply the decrement operation for the current query
            for i in range(l, r + 1):
                nums[i] = max(0, nums[i] - val)
            
            # Check if the array has become a zero array
            if all(x == 0 for x in nums):
                return k + 1  # Return the number of queries processed
        
        # If no such k exists, return -1
        return -1