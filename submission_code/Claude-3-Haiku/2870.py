class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return -1
        
        max_length = 0
        start = 0
        
        while start < n:
            if start + 1 >= n:
                break
            
            if nums[start] + 1 == nums[start + 1]:
                end = start + 2
                while end < n and nums[end] - nums[end - 1] == (-1) ** (end - start):
                    end += 1
                max_length = max(max_length, end - start)
                start = end
            else:
                start += 1
        
        return max_length if max_length > 0 else -1