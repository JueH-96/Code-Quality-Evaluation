class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        total = sum(nums)
        
        # Calculate complete cycles and remainder
        cycles = target // total
        remainder = target % total
        
        # If remainder is 0, we only need complete cycles
        if remainder == 0:
            return n * cycles
        
        # Find shortest subarray with sum = remainder
        # Check in doubled array to handle wrap-around
        min_len = float('inf')
        extended = nums + nums
        
        # Sliding window approach
        left = 0
        window_sum = 0
        
        for right in range(len(extended)):
            window_sum += extended[right]
            
            # Shrink window while sum is too large
            while window_sum > remainder and left <= right:
                window_sum -= extended[left]
                left += 1
            
            # Check if we found valid subarray
            if window_sum == remainder:
                min_len = min(min_len, right - left + 1)
        
        # Return -1 if no valid subarray found
        if min_len == float('inf'):
            return -1
        
        # Return total length including complete cycles
        return cycles * n + min_len