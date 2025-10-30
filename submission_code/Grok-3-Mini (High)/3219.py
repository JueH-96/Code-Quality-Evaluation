from typing import List

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        if n <= 1:
            return nums
        
        # Sort indices based on the values in nums
        sorted_idx = sorted(range(n), key=lambda i: nums[i])
        
        # Find connected components where consecutive elements in sorted order have diff <= limit
        components = []
        current_comp = [sorted_idx[0]]
        for i in range(1, n):
            if abs(nums[sorted_idx[i]] - nums[sorted_idx[i - 1]]) <= limit:
                current_comp.append(sorted_idx[i])
            else:
                components.append(current_comp)
                current_comp = [sorted_idx[i]]
        components.append(current_comp)  # Add the last component
        
        # For each component, sort the values and indices, and assign sorted values to sorted indices
        for comp in components:
            # Extract values for the component
            values = [nums[idx] for idx in comp]
            # Sort the values
            sorted_values = sorted(values)
            # Sort the indices by position
            sorted_indices = sorted(comp)
            # Assign the sorted values to the sorted indices in nums
            for i, idx in enumerate(sorted_indices):
                nums[idx] = sorted_values[i]
        
        return nums