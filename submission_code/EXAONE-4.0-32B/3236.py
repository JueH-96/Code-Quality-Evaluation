class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        n = len(nums)
        prefix_end = 0
        for i in range(1, n):
            if nums[i] == nums[i-1] + 1:
                prefix_end = i
            else:
                break
        
        total = sum(nums[:prefix_end+1])
        num_set = set(nums)
        
        x = total
        while x in num_set:
            x += 1
        return x