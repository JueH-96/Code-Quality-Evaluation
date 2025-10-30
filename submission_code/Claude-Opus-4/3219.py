class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        
        # Create pairs of (value, original_index) and sort by value
        indexed_nums = [(nums[i], i) for i in range(n)]
        indexed_nums.sort()
        
        # Group elements that can be swapped with each other
        groups = []
        current_group = [indexed_nums[0]]
        
        for i in range(1, n):
            # If current element is within limit of previous element, add to group
            if indexed_nums[i][0] - indexed_nums[i-1][0] <= limit:
                current_group.append(indexed_nums[i])
            else:
                # Start a new group
                groups.append(current_group)
                current_group = [indexed_nums[i]]
        
        # Don't forget the last group
        groups.append(current_group)
        
        # Create result array
        result = nums[:]
        
        # For each group, sort indices and assign sorted values
        for group in groups:
            # Extract values and indices
            values = [item[0] for item in group]
            indices = [item[1] for item in group]
            
            # Sort indices to determine where each value should go
            indices.sort()
            
            # Assign sorted values to sorted indices
            for i in range(len(group)):
                result[indices[i]] = values[i]
        
        return result