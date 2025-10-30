class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        even, odd = 1, 1
        max_len = 1
        for i in range(1, len(nums)):
            if (nums[i] + nums[i-1]) % 2 == 0:
                even = odd + 1
                max_len = max(max_len, even)
            else:
                odd = even + 1
                max_len = max(max_len, odd)
        return max_len