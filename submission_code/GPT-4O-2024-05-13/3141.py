class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        total_sum = sum(nums)
        
        # If the total sum of nums is greater than or equal to target, we can solve it within one cycle
        if total_sum >= target:
            prefix_sum = 0
            min_length = float('inf')
            left = 0
            
            for right in range(n):
                prefix_sum += nums[right]
                
                while prefix_sum >= target:
                    min_length = min(min_length, right - left + 1)
                    prefix_sum -= nums[left]
                    left += 1
            
            return min_length if min_length != float('inf') else -1
        
        # If the total sum of nums is less than target, we need to consider multiple cycles
        k = target // total_sum
        remainder = target % total_sum
        
        if remainder == 0:
            return k * n
        
        prefix_sum = 0
        min_length = float('inf')
        left = 0
        
        for right in range(2 * n):
            prefix_sum += nums[right % n]
            
            while prefix_sum >= remainder:
                min_length = min(min_length, right - left + 1)
                prefix_sum -= nums[left % n]
                left += 1
        
        return min_length + k * n if min_length != float('inf') else -1