class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        min_length = float('inf')
        curr_sum = 0
        left = 0
        
        for right in range(n):
            curr_sum += nums[right]
            
            while curr_sum >= target:
                min_length = min(min_length, right - left + 1)
                curr_sum -= nums[left]
                left += 1
        
        return min_length if min_length != float('inf') else -1