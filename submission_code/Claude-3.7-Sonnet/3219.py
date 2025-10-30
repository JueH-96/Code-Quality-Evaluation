class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        
        # Create pairs of (value, original_index)
        pairs = [(nums[i], i) for i in range(n)]
        
        # Sort by value
        pairs.sort()
        
        # Identify groups where consecutive elements have diff ≤ limit
        groups = []
        current_group = [pairs[0]]
        
        for i in range(1, n):
            if pairs[i][0] - pairs[i-1][0] <= limit:
                current_group.append(pairs[i])
            else:
                groups.append(current_group)
                current_group = [pairs[i]]
        
        groups.append(current_group)  # Add the last group
        
        # Rearrange the array
        result = nums.copy()
        
        for group in groups:
            values = [pair[0] for pair in group]
            indices = [pair[1] for pair in group]
            
            # Sort indices to get positions in ascending order
            indices.sort()
            
            # Assign sorted values to sorted positions
            for i in range(len(group)):
                result[indices[i]] = values[i]
        
        return result