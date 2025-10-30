class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        max_len = 0
        for num in set(nums):
            l = 0
            count = 0
            for r in range(len(nums)):
                if nums[r] != num:
                    count += 1
                while count > k:
                    if nums[l] != num:
                        count -= 1
                    l += 1
                max_len = max(max_len, r - l + 1)
        return max_len