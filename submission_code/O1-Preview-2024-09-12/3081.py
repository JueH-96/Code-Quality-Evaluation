class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        max_freq = curr_freq = 1
        for i in range(1, n):
            if nums[i] == nums[i-1]:
                curr_freq += 1
                if curr_freq > max_freq:
                    max_freq = curr_freq
            else:
                curr_freq = 1
        if max_freq > n // 2:
            return 2 * max_freq - n
        else:
            return n % 2