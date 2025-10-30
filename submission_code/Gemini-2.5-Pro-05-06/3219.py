from typing import List

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        if n == 0:
            return []

        # Create pairs of (value, original_index)
        # Example: if nums = [1,5,3], indexed_nums will be [(1,0), (5,1), (3,2)]
        indexed_nums = []
        for i in range(n):
            indexed_nums.append((nums[i], i))

        # Sort these pairs based on value.
        # Python's default tuple sort uses the first element, then second, etc.
        # So, (value, original_index) pairs are sorted primarily by value.
        # Example: for [(1,0), (5,1), (3,2)], sorted version is [(1,0), (3,2), (5,1)]
        indexed_nums.sort()

        result_array = [0] * n  # To store the final constructed array
        
        i = 0  # Pointer to the start of the current component in indexed_nums
        while i < n:
            # j will be the pointer to the element AFTER the end of the current component
            j = i + 1
            # Extend the current component:
            # Add indexed_nums[j] to the component if its value is close enough
            # to the value of the previous element in the component (indexed_nums[j-1]).
            while j < n and indexed_nums[j][0] - indexed_nums[j-1][0] <= limit:
                j += 1
            
            # The current component's elements in indexed_nums are from index i to j-1.
            # These elements form a group whose values can be permuted among their original positions.
            
            # Gather all original indices from this component.
            # Example: if component is [(1,0), (3,2), (5,1)], indices are [0,2,1]
            component_original_indices = [indexed_nums[k][1] for k in range(i, j)]
            
            # Gather all values from this component.
            # These values are already sorted as they come from the sorted indexed_nums list.
            # Example: values are [1,3,5]
            component_values = [indexed_nums[k][0] for k in range(i, j)]
            
            # Sort the original indices. This is crucial for placing the smallest values
            # of the component into the smallest available original positions,
            # ensuring the lexicographically smallest result.
            # Example: sorted indices for [0,2,1] become [0,1,2]
            component_original_indices.sort()
            
            # Place the (sorted) values into the (now sorted) original index positions.
            # Example: result_array[0]=1, result_array[1]=3, result_array[2]=5
            for k in range(len(component_values)):
                idx_to_fill = component_original_indices[k]
                val_to_place = component_values[k]
                result_array[idx_to_fill] = val_to_place
            
            # Move pointer i to the start of the next component.
            i = j
            
        return result_array