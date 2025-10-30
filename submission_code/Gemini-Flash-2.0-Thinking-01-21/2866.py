class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_length = 0
        n = len(nums)
        for l in range(n):
            if nums[l] % 2 == 0 and nums[l] <= threshold:
                current_length = 1
                for r in range(l + 1, n):
                    if nums[r] <= threshold and nums[r] % 2 != nums[r - 1] % 2:
                        current_length += 1
                    else:
                        break
                max_length = max(max_length, current_length)
        return max_length