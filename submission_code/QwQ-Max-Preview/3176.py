from typing import List

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        min_sum = float('inf')
        n = len(nums)
        for j in range(1, n - 1):
            # Find the minimum valid left element
            min_left = None
            for i in range(j):
                if nums[i] < nums[j]:
                    if min_left is None or nums[i] < min_left:
                        min_left = nums[i]
            # Find the minimum valid right element
            min_right = None
            for k in range(j + 1, n):
                if nums[k] < nums[j]:
                    if min_right is None or nums[k] < min_right:
                        min_right = nums[k]
            # Calculate the sum if both elements exist
            if min_left is not None and min_right is not None:
                current_sum = min_left + nums[j] + min_right
                if current_sum < min_sum:
                    min_sum = current_sum
        return min_sum if min_sum != float('inf') else -1