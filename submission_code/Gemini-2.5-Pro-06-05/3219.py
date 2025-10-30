from typing import List

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        # Pair each number with its original index
        indexed_nums = [(nums[i], i) for i in range(n)]
        
        # Sort by value to easily find swappable groups
        indexed_nums.sort()

        result = [0] * n
        i = 0
        while i < n:
            j = i
            # Find the end of the current swappable group (component)
            while j + 1 < n and indexed_nums[j+1][0] - indexed_nums[j][0] <= limit:
                j += 1
            
            # The elements from indexed_nums[i...j] form a component where values
            # can be freely permuted among their original positions.
            component_slice = indexed_nums[i : j+1]
            
            # Extract the original indices and values from this component
            component_indices = [p[1] for p in component_slice]
            component_values = [p[0] for p in component_slice] # These are already sorted
            
            # To get the lexicographically smallest permutation, we sort the
            # original indices and assign the sorted values to these sorted positions.
            component_indices.sort()
            
            # Assign the sorted values to the sorted indices
            for idx, val in zip(component_indices, component_values):
                result[idx] = val
                
            # Move to the start of the next component
            i = j + 1
            
        return result