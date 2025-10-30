class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            current_max = nums[i]
            for j in range(i, n):
                current_max = max(current_max, nums[j])
                if current_max == nums[i] and current_max == nums[j]:
                    count += 1
        
        return count