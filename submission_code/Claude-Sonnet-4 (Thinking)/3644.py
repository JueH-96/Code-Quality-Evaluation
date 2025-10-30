class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        min_sum = float('inf')
        n = len(nums)
        
        for length in range(l, r + 1):
            # Calculate sum for the first window
            window_sum = sum(nums[:length])
            if window_sum > 0:
                min_sum = min(min_sum, window_sum)
            
            # Slide the window
            for i in range(length, n):
                window_sum = window_sum - nums[i - length] + nums[i]
                if window_sum > 0:
                    min_sum = min(min_sum, window_sum)
        
        return min_sum if min_sum != float('inf') else -1