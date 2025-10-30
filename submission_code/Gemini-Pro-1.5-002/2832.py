class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_len = 0
        for num in set(nums):
            left = 0
            count = 0
            for right in range(n):
                if nums[right] != num:
                    count += 1
                while count > k:
                    if nums[left] != num:
                        count -= 1
                    left += 1
                max_len = max(max_len, right - left + 1)
        return max_len