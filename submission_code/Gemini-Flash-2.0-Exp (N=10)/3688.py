class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def kadane(arr):
            max_so_far = float('-inf')
            current_max = 0
            for num in arr:
                current_max = max(num, current_max + num)
                max_so_far = max(max_so_far, current_max)
            return max_so_far

        max_sum = kadane(nums)
        unique_nums = set(nums)
        for num_to_remove in unique_nums:
            temp_arr = [x for x in nums if x != num_to_remove]
            if temp_arr:
                max_sum = max(max_sum, kadane(temp_arr))
        
        return max_sum