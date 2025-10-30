class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        if not nums:
            return []
        
        # Create a list of (value, index) tuples and sort by value
        sorted_with_index = sorted((v, i) for i, v in enumerate(nums))
        
        # Group the sorted list into consecutive groups where each step is <= limit
        groups = []
        if not sorted_with_index:
            return nums
        current_group = [sorted_with_index[0]]
        for i in range(1, len(sorted_with_index)):
            prev_v, prev_i = current_group[-1]
            curr_v, curr_i = sorted_with_index[i]
            if curr_v - prev_v <= limit:
                current_group.append((curr_v, curr_i))
            else:
                groups.append(current_group)
                current_group = [(curr_v, curr_i)]
        if current_group:
            groups.append(current_group)
        
        # Initialize the result as a copy of the original nums
        result = list(nums)
        
        # Process each group
        for group in groups:
            # Extract indices and values
            indices = [x[1] for x in group]
            values = [x[0] for x in group]
            
            # Sort the indices and the values
            sorted_indices = sorted(indices)
            sorted_values = sorted(values)
            
            # Assign the sorted values to the sorted indices
            for i in range(len(sorted_indices)):
                result[sorted_indices[i]] = sorted_values[i]
        
        return result