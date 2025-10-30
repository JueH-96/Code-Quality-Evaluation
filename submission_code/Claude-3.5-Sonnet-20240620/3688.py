class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def kadane(arr):
            max_sum = float('-inf')
            current_sum = 0
            for num in arr:
                current_sum = max(num, current_sum + num)
                max_sum = max(max_sum, current_sum)
            return max_sum

        n = len(nums)
        if n == 1:
            return nums[0]

        # Calculate the maximum subarray sum without removing any element
        max_sum_without_removal = kadane(nums)

        # Calculate prefix sums and suffix sums
        prefix_sum = [0] * n
        suffix_sum = [0] * n
        
        prefix_sum[0] = nums[0]
        for i in range(1, n):
            prefix_sum[i] = prefix_sum[i-1] + nums[i]
        
        suffix_sum[-1] = nums[-1]
        for i in range(n-2, -1, -1):
            suffix_sum[i] = suffix_sum[i+1] + nums[i]

        # Calculate the maximum subarray sum after removing one element
        max_sum_with_removal = float('-inf')
        unique_nums = set(nums)
        
        for x in unique_nums:
            indices = [i for i in range(n) if nums[i] == x]
            if len(indices) == n:
                continue
            
            if indices[0] > 0:
                left_sum = prefix_sum[indices[0] - 1]
            else:
                left_sum = 0
            
            if indices[-1] < n - 1:
                right_sum = suffix_sum[indices[-1] + 1]
            else:
                right_sum = 0
            
            max_sum_with_removal = max(max_sum_with_removal, left_sum + right_sum)

        return max(max_sum_without_removal, max_sum_with_removal)