class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        max_len = 1
        n = len(nums)
        
        # Check for strictly increasing subarrays
        for i in range(n):
            current_len = 1
            for j in range(i+1, n):
                if nums[j] > nums[j-1]:
                    current_len += 1
                else:
                    break
            if current_len > max_len:
                max_len = current_len
        
        # Check for strictly decreasing subarrays
        for i in range(n):
            current_len = 1
            for j in range(i+1, n):
                if nums[j] < nums[j-1]:
                    current_len += 1
                else:
                    break
            if current_len > max_len:
                max_len = current_len
        
        return max_len