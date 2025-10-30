class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        total = sum(nums)
        
        # If target is larger than sum of array
        if target > total:
            # Get complete cycles needed
            complete_cycles = target // total
            # Reduce target by sum of complete cycles
            remaining = target % total
            if remaining == 0:
                return n * complete_cycles
        else:
            complete_cycles = 0
            remaining = target
            
        # Create double array for sliding window
        nums = nums + nums
        curr_sum = 0
        left = 0
        min_len = float('inf')
        
        # Sliding window to find subarray with sum = remaining
        for right in range(len(nums)):
            curr_sum += nums[right]
            
            while curr_sum > remaining:
                curr_sum -= nums[left]
                left += 1
                
            if curr_sum == remaining:
                min_len = min(min_len, right - left + 1)
                
        if min_len == float('inf'):
            return -1
            
        return min_len + (complete_cycles * n)