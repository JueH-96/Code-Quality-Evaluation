class Solution:
    def countSubarrays(self, nums, k):
        count = 0
        for i in range(len(nums)):
            and_val = nums[i]
            if and_val == k:
                count += 1
            for j in range(i+1, len(nums)):
                and_val &= nums[j]
                if and_val == k:
                    count += 1
                if and_val < k:
                    break
        return count