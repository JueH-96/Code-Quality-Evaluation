class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        unique_values = set(nums)
        overall_max_len = 0
        for val in unique_values:
            left = 0
            max_len = 0
            deletions = 0
            count_val = 0
            for right in range(len(nums)):
                if nums[right] == val:
                    count_val += 1
                else:
                    deletions += 1
                while deletions > k:
                    if nums[left] != val:
                        deletions -= 1
                    else:
                        count_val -= 1
                    left += 1
                max_len = max(max_len, count_val)
            overall_max_len = max(overall_max_len, max_len)
        return overall_max_len