class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        def count_continuous_subarrays(start, end):
            return (end - start + 1) * (end - start + 2) // 2

        total_continuous_subarrays = 0
        start = 0
        
        for end in range(len(nums)):
            while start < end and not (nums[start] <= nums[end] <= nums[start] + 2):
                start += 1
            total_continuous_subarrays += count_continuous_subarrays(start, end)
        
        return total_continuous_subarrays