class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        
        # Create (value, index) pairs and sort by value
        indexed_nums = [(nums[i], i) for i in range(n)]
        indexed_nums.sort()
        
        # Group elements that can be swapped
        groups = []
        current_group = [indexed_nums[0]]
        
        for i in range(1, n):
            if indexed_nums[i][0] - indexed_nums[i-1][0] <= limit:
                current_group.append(indexed_nums[i])
            else:
                groups.append(current_group)
                current_group = [indexed_nums[i]]
        
        groups.append(current_group)
        
        # For each group, place elements in ascending order at the positions
        result = [0] * n
        for group in groups:
            # Extract values and indices
            values = [item[0] for item in group]
            indices = [item[1] for item in group]
            
            # Sort indices to place smallest values at leftmost positions
            indices.sort()
            
            # Place sorted values at sorted indices
            for i in range(len(group)):
                result[indices[i]] = values[i]
        
        return result