class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        # Pair each element with its index and sort by the element value
        sorted_vals_indices = sorted(zip(nums, range(n)), key=lambda x: x[0])
        
        groups = []
        if not sorted_vals_indices:
            return []
        
        # Group consecutive elements with differences <= limit
        current_group = [sorted_vals_indices[0]]
        for i in range(1, n):
            current_val, current_idx = sorted_vals_indices[i]
            prev_val, prev_idx = sorted_vals_indices[i-1]
            if current_val - prev_val <= limit:
                current_group.append((current_val, current_idx))
            else:
                groups.append(current_group)
                current_group = [(current_val, current_idx)]
        groups.append(current_group)  # Add the last group
        
        # Prepare the result array
        res = [0] * n
        for group in groups:
            # Extract values and indices from the group
            values = [val for val, idx in group]
            indices = [idx for val, idx in group]
            # Sort the indices to assign the smallest value to the earliest index
            sorted_indices = sorted(indices)
            # Assign sorted values to the sorted indices
            for i in range(len(sorted_indices)):
                res[sorted_indices[i]] = values[i]
        
        return res