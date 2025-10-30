class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        n = len(nums)
        length = 1
        for i in range(1, n):
            if nums[i] == nums[i - 1] + 1:
                length += 1
            else:
                break
        
        prefix_sum = sum(nums[:length])
        
        s = set(nums)
        
        while prefix_sum in s:
            prefix_sum += 1
        
        return prefix_sum