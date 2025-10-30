class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        max_len = 1
        
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                
                if len(subarray) <= 1:
                    max_len = max(max_len, len(subarray))
                    continue
                
                is_increasing = True
                for k in range(len(subarray) - 1):
                    if subarray[k] >= subarray[k+1]:
                        is_increasing = False
                        break
                
                if is_increasing:
                    max_len = max(max_len, len(subarray))
                    continue
                
                is_decreasing = True
                for k in range(len(subarray) - 1):
                    if subarray[k] <= subarray[k+1]:
                        is_decreasing = False
                        break
                
                if is_decreasing:
                    max_len = max(max_len, len(subarray))
        
        return max_len