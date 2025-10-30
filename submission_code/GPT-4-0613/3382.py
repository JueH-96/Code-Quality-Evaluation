class Solution:
    def numberOfSubarrays(self, nums):
        count = 0
        for i in range(len(nums)):
            max_num = nums[i]
            for j in range(i, len(nums)):
                max_num = max(max_num, nums[j])
                if nums[i] == max_num and nums[j] == max_num:
                    count += 1
        return count