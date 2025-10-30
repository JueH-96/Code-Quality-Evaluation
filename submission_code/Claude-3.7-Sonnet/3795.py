class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        # Make a copy of the array to modify
        nums_copy = nums.copy()
        
        for k, (l, r, val) in enumerate(queries):
            # Process each query
            for i in range(l, r + 1):
                # Only decrement if the value won't become negative
                if nums_copy[i] >= val:
                    nums_copy[i] -= val
            
            # Check if the array is a Zero Array after this query
            if all(x == 0 for x in nums_copy):
                return k + 1
        
        # If we've processed all queries and the array isn't a Zero Array
        return -1