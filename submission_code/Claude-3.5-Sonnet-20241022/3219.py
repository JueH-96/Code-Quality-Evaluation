class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        # Create array of (num, index) pairs
        pairs = [(num, i) for i, num in enumerate(nums)]
        # Sort by values
        pairs.sort()
        
        # Group numbers that can be swapped with each other
        groups = []
        curr_group = [pairs[0]]
        
        for i in range(1, n):
            # If difference between consecutive numbers <= limit
            # They can be part of same group
            if pairs[i][0] - pairs[i-1][0] <= limit:
                curr_group.append(pairs[i])
            else:
                groups.append(curr_group)
                curr_group = [pairs[i]]
        groups.append(curr_group)
        
        # For each group, sort the indices and assign values optimally
        result = nums[:]
        for group in groups:
            values = sorted([pair[0] for pair in group])
            indices = sorted([pair[1] for pair in group])
            
            # Assign smallest values to smallest indices
            for val, idx in zip(values, indices):
                result[idx] = val
                
        return result