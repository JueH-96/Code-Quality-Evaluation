import bisect

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        n = len(nums)
        max_freq = 0
        i = 0
        while i < n:
            t = nums[i]
            # Find the original count of t in nums
            left = bisect.bisect_left(nums, t)
            right = bisect.bisect_right(nums, t)
            original_count = right - left
            
            # Determine the range [t-k, t+k]
            lower = t - k
            upper = t + k
            l = bisect.bisect_left(nums, lower)
            r = bisect.bisect_right(nums, upper)
            total_in_range = r - l
            
            available = total_in_range - original_count
            additional = min(numOperations, available)
            current_freq = original_count + additional
            
            if current_freq > max_freq:
                max_freq = current_freq
            
            # Move to the next unique element
            i = right
        
        return max_freq