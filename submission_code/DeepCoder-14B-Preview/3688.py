class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def kadane(arr):
            if not arr:
                return 0
            max_current = max_global = arr[0]
            for num in arr[1:]:
                max_current = max(num, max_current + num)
                max_global = max(max_global, max_current)
            return max_global
        
        max_sum = kadane(nums)
        
        unique_x = set(nums)
        for x in unique_x:
            new_array = [num for num in nums if num != x]
            if not new_array:
                continue
            current_max = kadane(new_array)
            if current_max > max_sum:
                max_sum = current_max
        
        return max_sum