class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_length = 1
        
        for i in range(n):
            # Check for strictly increasing subarray
            increasing_length = 1
            for j in range(i+1, n):
                if nums[j] > nums[j-1]:
                    increasing_length += 1
                else:
                    break
            max_length = max(max_length, increasing_length)
            
            # Check for strictly decreasing subarray
            decreasing_length = 1
            for j in range(i+1, n):
                if nums[j] < nums[j-1]:
                    decreasing_length += 1
                else:
                    break
            max_length = max(max_length, decreasing_length)
        
        return max_length