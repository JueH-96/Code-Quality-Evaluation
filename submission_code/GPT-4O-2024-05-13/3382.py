class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            max_val = nums[i]
            for j in range(i, n):
                max_val = max(max_val, nums[j])
                if nums[i] == max_val and nums[j] == max_val:
                    count += 1
        
        return count