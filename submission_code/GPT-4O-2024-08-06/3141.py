class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        total_sum = sum(nums)
        n = len(nums)
        
        # If the total sum of nums is greater than or equal to target, we can find a subarray in one cycle
        if total_sum >= target:
            # Use a sliding window to find the shortest subarray in one cycle
            left = 0
            current_sum = 0
            min_length = float('inf')
            
            for right in range(n):
                current_sum += nums[right]
                
                while current_sum >= target:
                    min_length = min(min_length, right - left + 1)
                    current_sum -= nums[left]
                    left += 1
            
            return min_length if min_length != float('inf') else -1
        
        # If total_sum < target, we need to consider multiple cycles
        # Calculate how many full cycles we need to reach at least target
        full_cycles = target // total_sum
        remaining_target = target % total_sum
        
        # If remaining_target is 0, we can achieve the target with full_cycles * n elements
        if remaining_target == 0:
            return full_cycles * n
        
        # Otherwise, we need to find the shortest subarray in one cycle that sums to remaining_target
        left = 0
        current_sum = 0
        min_length = float('inf')
        
        for right in range(2 * n):  # We consider two cycles to handle wrap-around
            current_sum += nums[right % n]
            
            while current_sum >= remaining_target:
                min_length = min(min_length, right - left + 1)
                current_sum -= nums[left % n]
                left += 1
        
        # If we found a valid subarray, return the total length considering full cycles
        if min_length != float('inf'):
            return full_cycles * n + min_length
        
        # If no valid subarray was found, return -1
        return -1