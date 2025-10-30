class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def max_subarray(arr):
            max_so_far = float('-inf')
            current_max = 0
            for x in arr:
                current_max += x
                if max_so_far < current_max:
                    max_so_far = current_max
                if current_max < 0:
                    current_max = 0
            return max_so_far if max_so_far != float('-inf') else max(arr)

        max_sum = max_subarray(nums)
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1

        for x in set(nums):
            new_nums = [num for num in nums if num != x]
            if not new_nums:
                continue
            max_sum = max(max_sum, max_subarray(new_nums))

        return max_sum