class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_length = 0
        n = len(nums)

        for start in range(n):
            if nums[start] % 2 == 0 and nums[start] <= threshold:
                current_length = 1
                for i in range(start, n - 1):
                    if nums[i + 1] <= threshold and nums[i] % 2 != nums[i + 1] % 2:
                        current_length += 1
                    else:
                        break
                max_length = max(max_length, current_length)

        return max_length