class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        max_len = 0
        distinct_nums = set(nums)
        for target_num in distinct_nums:
            window_start = 0
            count = 0
            for window_end in range(len(nums)):
                if nums[window_end] == target_num:
                    count += 1
                elements_to_remove = (window_end - window_start + 1) - count
                while elements_to_remove > k:
                    if nums[window_start] == target_num:
                        count -= 1
                    window_start += 1
                    elements_to_remove = (window_end - window_start + 1) - count
                max_len = max(max_len, count)
        return max_len