class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        # Create a list of tuples (original index, value) and sort by value
        sorted_with_indices = sorted(enumerate(nums), key=lambda x: x[1])
        groups = []
        if not sorted_with_indices:
            return nums
        
        # Form groups where each consecutive pair has a difference <= limit
        current_group = [sorted_with_indices[0]]
        prev_val = current_group[0][1]
        for i in range(1, len(sorted_with_indices)):
            val = sorted_with_indices[i][1]
            if val - prev_val <= limit:
                current_group.append(sorted_with_indices[i])
            else:
                groups.append(current_group)
                current_group = [sorted_with_indices[i]]
                prev_val = val
        # Add the last group
        groups.append(current_group)
        
        # Process each group to assign sorted values to sorted original indices
        result = list(nums)  # Create a copy to modify
        for group in groups:
            original_indices = [x[0] for x in group]
            original_indices.sort()
            values = [x[1] for x in group]
            values.sort()
            for i in range(len(original_indices)):
                result[original_indices[i]] = values[i]
        
        return result