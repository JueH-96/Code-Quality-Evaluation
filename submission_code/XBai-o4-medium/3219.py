class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        # Create a list of (value, index) pairs and sort them based on the value
        sorted_with_indices = sorted([(nums[i], i) for i in range(len(nums))], key=lambda x: x[0])
        
        # Group the sorted elements into connected components based on the limit
        groups = []
        if not sorted_with_indices:
            return []
        current_group = [sorted_with_indices[0]]
        for i in range(1, len(sorted_with_indices)):
            prev_val, prev_idx = sorted_with_indices[i-1]
            curr_val, curr_idx = sorted_with_indices[i]
            if curr_val - prev_val <= limit:
                current_group.append(sorted_with_indices[i])
            else:
                groups.append(current_group)
                current_group = [sorted_with_indices[i]]
        groups.append(current_group)
        
        # Prepare the result by assigning sorted values to sorted indices within each group
        res = [0] * len(nums)
        for group in groups:
            values = []
            indices = []
            for (val, idx) in group:
                values.append(val)
                indices.append(idx)
            indices.sort()
            for i in range(len(values)):
                res[indices[i]] = values[i]
        
        return res