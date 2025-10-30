class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        total_sum = sum(nums)
        
        # Handle the case where we need multiple complete cycles
        complete_cycles = target // total_sum
        remainder = target % total_sum
        
        # If remainder is 0, we just need complete cycles
        if remainder == 0:
            return complete_cycles * n
        
        # Create a doubled array to handle wrap-around cases
        doubled_nums = nums + nums
        
        # Use sliding window to find shortest subarray with sum = remainder
        min_length = float('inf')
        left = 0
        current_sum = 0
        
        for right in range(len(doubled_nums)):
            current_sum += doubled_nums[right]
            
            # Shrink window while sum is too large
            while current_sum > remainder and left <= right:
                current_sum -= doubled_nums[left]
                left += 1
            
            # Check if we found a valid subarray
            if current_sum == remainder:
                min_length = min(min_length, right - left + 1)
        
        # If no valid subarray found
        if min_length == float('inf'):
            return -1
        
        # Total length is complete cycles plus the remainder subarray
        return complete_cycles * n + min_length