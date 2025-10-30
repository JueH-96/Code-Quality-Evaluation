class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        total_sum = sum(nums)
        
        # If target is divisible by total_sum, we need exactly target//total_sum cycles
        if target % total_sum == 0:
            return (target // total_sum) * n
        
        # We need at most 2 cycles to find any possible subarray
        # because any subarray spanning more than 2 cycles can be optimized
        extended_nums = nums + nums
        
        # Find the shortest subarray with sum equal to target % total_sum
        remainder = target % total_sum
        min_len = float('inf')
        
        # Use sliding window to find shortest subarray with sum = remainder
        left = 0
        current_sum = 0
        
        for right in range(len(extended_nums)):
            current_sum += extended_nums[right]
            
            # Shrink window from left while sum is greater than remainder
            while current_sum > remainder and left <= right:
                current_sum -= extended_nums[left]
                left += 1
            
            # Check if we found a valid subarray
            if current_sum == remainder:
                min_len = min(min_len, right - left + 1)
        
        # If no subarray found with the remainder sum, return -1
        if min_len == float('inf'):
            return -1
        
        # Calculate total length: complete cycles + shortest subarray for remainder
        complete_cycles = target // total_sum
        return complete_cycles * n + min_len