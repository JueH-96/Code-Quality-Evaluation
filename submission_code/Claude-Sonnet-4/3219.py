class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        
        # Create list of (value, original_index) and sort by value
        indexed_nums = [(nums[i], i) for i in range(n)]
        indexed_nums.sort()
        
        result = [0] * n
        i = 0
        
        while i < n:
            # Find the end of current group (elements that can be swapped with each other)
            j = i
            while j + 1 < n and indexed_nums[j + 1][0] - indexed_nums[j][0] <= limit:
                j += 1
            
            # Extract values and indices for current group
            group_values = [indexed_nums[k][0] for k in range(i, j + 1)]
            group_indices = [indexed_nums[k][1] for k in range(i, j + 1)]
            
            # Sort indices to place smallest values in earliest positions
            group_indices.sort()
            
            # Assign sorted values to sorted positions
            for k in range(len(group_values)):
                result[group_indices[k]] = group_values[k]
            
            i = j + 1
        
        return result