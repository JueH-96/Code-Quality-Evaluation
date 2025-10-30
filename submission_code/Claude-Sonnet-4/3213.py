class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_val = max(nums)
        n = len(nums)
        
        # Find positions of maximum elements
        max_positions = []
        for i in range(n):
            if nums[i] == max_val:
                max_positions.append(i)
        
        if len(max_positions) < k:
            return 0
        
        count = 0
        
        # For each valid window of k maximum elements
        for i in range(len(max_positions) - k + 1):
            # Window boundaries
            window_start = max_positions[i]
            window_end = max_positions[i + k - 1]
            
            # Left boundary for subarrays
            left_boundary = -1 if i == 0 else max_positions[i - 1]
            
            # Right boundary for subarrays  
            right_boundary = n if i + k == len(max_positions) else max_positions[i + k]
            
            # Count valid subarrays
            left_choices = window_start - left_boundary
            right_choices = right_boundary - window_end
            
            count += left_choices * right_choices
        
        return count