class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_len = 0
        for i in range(len(nums)):
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                curr_len = 1
                curr_max = nums[i]
                for j in range(i + 1, len(nums)):
                    if nums[j] <= threshold and nums[j] % 2 != nums[j - 1] % 2:
                        curr_len += 1
                        curr_max = nums[j]
                    else:
                        break
                max_len = max(max_len, curr_len)
        return max_len