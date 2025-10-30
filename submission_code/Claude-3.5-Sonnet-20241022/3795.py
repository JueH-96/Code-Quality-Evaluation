class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        # Function to check if array is zero array
        def isZeroArray(arr):
            return all(x == 0 for x in arr)
        
        # Function to process a single query
        def processQuery(arr, l, r, val, selected):
            # Create a copy of the array
            new_arr = arr.copy()
            # Decrement values at selected indices
            for i in range(l, r + 1):
                if selected[i - l]:
                    new_arr[i] = max(0, new_arr[i] - val)
            return new_arr
        
        n = len(nums)
        k = 0
        
        # Try each query in sequence
        for k in range(len(queries)):
            l, r, val = queries[k]
            length = r - l + 1
            
            # Try all possible combinations of index selections
            found_zero = False
            # Generate all possible combinations using binary numbers
            for mask in range(1 << length):
                # Convert mask to selection array
                selected = [(mask >> i) & 1 for i in range(length)]
                
                # Create a copy of nums and process all queries up to k
                current = nums.copy()
                for i in range(k + 1):
                    ql, qr, qval = queries[i]
                    if i == k:
                        # For the last query, use our selected indices
                        current = processQuery(current, ql, qr, qval, selected)
                    else:
                        # For previous queries, select all indices
                        current = processQuery(current, ql, qr, qval, [1] * (qr - ql + 1))
                
                if isZeroArray(current):
                    return k + 1
        
        return -1