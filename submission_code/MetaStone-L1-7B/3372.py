class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        current_incr = 1
        current_decr = 1
        max_len = 1
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                current_incr += 1
                current_decr = 1
            elif nums[i] < nums[i-1]:
                current_decr += 1
                current_incr = 1
            else:
                current_incr = 1
                current_decr = 1
            current_max = max(current_incr, current_decr)
            if current_max > max_len:
                max_len = current_max
        return max_len