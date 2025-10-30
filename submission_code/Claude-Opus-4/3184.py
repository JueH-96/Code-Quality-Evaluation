class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Transform the array: nums[i] - i
        transformed = [(nums[i] - i, i) for i in range(n)]
        
        # Sort by transformed value, keeping track of original indices
        sorted_transformed = sorted(transformed)
        
        # Coordinate compression
        coord_map = {}
        coord = 0
        for i in range(n):
            if i == 0 or sorted_transformed[i][0] != sorted_transformed[i-1][0]:
                coord_map[sorted_transformed[i][0]] = coord
                coord += 1
        
        # Use a segment tree or BIT to maintain maximum values
        # For simplicity, using a dictionary to track maximum sum ending at each compressed coordinate
        max_sum_at = {}
        
        result = float('-inf')
        
        # Process elements in original order
        for i in range(n):
            val = nums[i] - i
            compressed_val = coord_map[val]
            
            # Find maximum sum of subsequences ending at positions j < i 
            # where transformed[j] <= transformed[i]
            max_prev = 0
            for cv in range(compressed_val + 1):
                if cv in max_sum_at:
                    max_prev = max(max_prev, max_sum_at[cv])
            
            # Current maximum sum ending at position i
            current_sum = max_prev + nums[i]
            
            # Update the maximum sum at this compressed coordinate
            if compressed_val not in max_sum_at:
                max_sum_at[compressed_val] = current_sum
            else:
                max_sum_at[compressed_val] = max(max_sum_at[compressed_val], current_sum)
            
            result = max(result, current_sum)
        
        return result