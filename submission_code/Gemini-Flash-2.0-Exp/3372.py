class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        max_len = 1
        
        for i in range(n):
            # Increasing
            curr_len = 1
            for j in range(i + 1, n):
                if nums[j] > nums[j-1]:
                    curr_len += 1
                else:
                    break
            max_len = max(max_len, curr_len)
            
            # Decreasing
            curr_len = 1
            for j in range(i + 1, n):
                if nums[j] < nums[j-1]:
                    curr_len += 1
                else:
                    break
            max_len = max(max_len, curr_len)
            
        return max_len