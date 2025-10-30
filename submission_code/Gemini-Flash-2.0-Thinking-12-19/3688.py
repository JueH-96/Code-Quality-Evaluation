class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def get_max_subarray_sum(arr):
            if not arr:
                return -float('inf')
            max_so_far = -float('inf')
            current_max = 0
            for x in arr:
                current_max += x
                if current_max > max_so_far:
                    max_so_far = current_max
                if current_max < 0:
                    current_max = 0
            if max_so_far == -float('inf'):
                return max(arr) if arr else -float('inf')
            return max_so_far if max_so_far != -float('inf') else max(arr)

        max_sum_overall = get_max_subarray_sum(nums)
        unique_nums = set(nums)

        for x in unique_nums:
            nums_removed_x = [num for num in nums if num != x]
            if nums_removed_x:
                max_sum_removed_x = get_max_subarray_sum(nums_removed_x)
                max_sum_overall = max(max_sum_overall, max_sum_removed_x)
            elif len(nums) > 1: # handle edge case where removing x makes the array empty, but original is not empty.
                pass # should not happen as per problem description, but to be safe.
            else: # nums has only one element, and we remove it, array becomes empty, which is not allowed.
                pass # should not happen as per problem description, but to be safe.

        return max_sum_overall