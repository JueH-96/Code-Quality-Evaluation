class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_len = 0
        for i in range(len(nums)):
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                curr_len = 1
                for j in range(i + 1, len(nums)):
                    if nums[j] % 2 != nums[j - 1] % 2 and nums[j] <= threshold:
                        curr_len += 1
                    else:
                        break
                max_len = max(max_len, curr_len)
        return max_len