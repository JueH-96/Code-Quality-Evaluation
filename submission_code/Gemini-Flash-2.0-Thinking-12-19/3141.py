class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        extended_nums = nums * 2
        prefix_sums = [0] * (len(extended_nums) + 1)
        for i in range(len(extended_nums)):
            prefix_sums[i+1] = prefix_sums[i] + extended_nums[i]
        min_length = float('inf')
        found = False
        for start_index in range(len(extended_nums)):
            for end_index in range(start_index, len(extended_nums)):
                current_sum = prefix_sums[end_index + 1] - prefix_sums[start_index]
                if current_sum == target:
                    min_length = min(min_length, end_index - start_index + 1)
                    found = True
        if not found:
            return -1
        return min_length