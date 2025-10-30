class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_len = 0
        for i in range(n):
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                current_len = 1
                for j in range(i + 1, n):
                    if nums[j] <= threshold and nums[j] % 2 != nums[j - 1] % 2:
                        current_len += 1
                    else:
                        break
                max_len = max(max_len, current_len)
        return max_len