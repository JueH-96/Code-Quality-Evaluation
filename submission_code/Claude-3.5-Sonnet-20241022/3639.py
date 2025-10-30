class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        # Create a copy of nums to work with
        arr = nums.copy()
        n = len(arr)
        
        # Process each query
        for l, r in queries:
            # Find the minimum value in the range [l, r]
            min_val = min(arr[l:r+1])
            
            # If minimum value is 0, we can't decrement all elements in range
            # because we can't decrement elements selectively to make them equal
            if min_val == 0:
                # Decrement all non-zero elements by 1
                for i in range(l, r+1):
                    if arr[i] > 0:
                        arr[i] -= 1
            else:
                # If minimum value is not 0, we can decrement all elements
                for i in range(l, r+1):
                    arr[i] -= 1
        
        # Check if array is a Zero Array
        return all(x == 0 for x in arr)