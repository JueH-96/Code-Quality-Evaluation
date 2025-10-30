class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_length = 0
        n = nums.length

        for i in range(n):
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                length = 1
                for j in range(i + 1, n):
                    if nums[j] <= threshold and nums[j] % 2 != nums[j - 1] % 2:
                        length += 1
                    else:
                        break
                max_length = max(max_length, length)

        return max_length