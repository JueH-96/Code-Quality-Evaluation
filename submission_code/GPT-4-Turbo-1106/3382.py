class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        count = 0
        for i in range(len(nums)):
            max_val = nums[i]
            for j in range(i, len(nums)):
                max_val = max(max_val, nums[j])
                if nums[i] == max_val and nums[j] == max_val:
                    count += 1
        return count