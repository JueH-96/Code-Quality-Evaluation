class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        max_length = 0
        unique_nums = sorted(list(set(nums)))

        for target in unique_nums:
            left = 0
            deletions = 0
            for right in range(len(nums)):
                if nums[right] != target:
                    deletions += 1

                while deletions > k:
                    if nums[left] != target:
                        deletions -= 1
                    left += 1

                equal_subarray_length = 0
                for i in range(left, right + 1):
                    if nums[i] == target:
                        equal_subarray_length += 1

                max_length = max(max_length, equal_subarray_length)

        return max_length