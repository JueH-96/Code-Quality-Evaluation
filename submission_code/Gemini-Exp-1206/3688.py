class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def kadane(arr):
            max_so_far = float('-inf')
            current_max = 0
            for i in range(len(arr)):
                current_max += arr[i]
                if max_so_far < current_max:
                    max_so_far = current_max
                if current_max < 0:
                    current_max = 0
            if max_so_far == float('-inf'):
                if not arr:
                    return float('-inf')
                else:
                    return max(arr)
            return max_so_far

        ans = kadane(nums)
        unique_nums = set(nums)
        for x in unique_nums:
            temp_nums = [num for num in nums if num != x]
            ans = max(ans, kadane(temp_nums))
        return ans