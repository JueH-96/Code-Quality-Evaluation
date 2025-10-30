class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def max_subarray_sum(arr):
            max_so_far = -float('inf')
            current_max = 0
            for num in arr:
                current_max += num
                if current_max > max_so_far:
                    max_so_far = current_max
                if current_max < 0:
                    current_max = 0
            return max_so_far

        max_sum = max_subarray_sum(nums)
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1

        for num in counts:
            temp_nums = []
            for val in nums:
                if val != num:
                    temp_nums.append(val)
            if temp_nums:
                max_sum = max(max_sum, max_subarray_sum(temp_nums))

        return max_sum