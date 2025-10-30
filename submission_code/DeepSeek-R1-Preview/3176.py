class Solution:
    def minimumSum(self, nums):
        n = len(nums)
        min_sum = float('inf')
        for j in range(1, n - 1):
            # Find left minimum
            left_min = None
            for i in range(j):
                if nums[i] < nums[j]:
                    if left_min is None or nums[i] < left_min:
                        left_min = nums[i]
            if left_min is None:
                continue
            # Find right minimum
            right_min = None
            for k in range(j + 1, n):
                if nums[k] < nums[j]:
                    if right_min is None or nums[k] < right_min:
                        right_min = nums[k]
            if right_min is None:
                continue
            # Update the minimum sum
            current_sum = left_min + nums[j] + right_min
            if current_sum < min_sum:
                min_sum = current_sum
        return min_sum if min_sum != float('inf') else -1