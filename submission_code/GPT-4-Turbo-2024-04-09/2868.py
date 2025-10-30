class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        total_continuous_subarrays = 0
        start = 0
        
        while start < n:
            end = start
            min_val = nums[start]
            max_val = nums[start]
            
            while end < n and max_val - min_val <= 2:
                total_continuous_subarrays += (end - start + 1)
                end += 1
                if end < n:
                    min_val = min(min_val, nums[end])
                    max_val = max(max_val, nums[end])
            
            start += 1
        
        return total_continuous_subarrays