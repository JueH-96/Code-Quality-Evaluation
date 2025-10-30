from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        min_val = float('inf')
        max_val = float('-inf')
        has_missing = False

        # Find the minimum and maximum values in the array
        for num in nums:
            if num == -1:
                has_missing = True
            else:
                min_val = min(min_val, num)
                max_val = max(max_val, num)

        # If there are no missing values, the difference is already minimized
        if not has_missing:
            max_diff = 0
            for i in range(1, n):
                max_diff = max(max_diff, abs(nums[i] - nums[i-1]))
            return max_diff

        # Initialize the optimal pair (x, y)
        x = min_val
        y = max_val

        # Replace missing values with x or y to minimize the maximum difference
        for i in range(n):
            if nums[i] == -1:
                if i > 0 and i < n-1:
                    prev = nums[i-1]
                    next = nums[i+1]
                    if prev != -1 and next != -1:
                        nums[i] = min(max(prev, next), y)
                    elif prev != -1:
                        nums[i] = min(max(prev, x), y)
                    elif next != -1:
                        nums[i] = min(max(next, x), y)
                    else:
                        nums[i] = x
                elif i > 0:
                    nums[i] = nums[i-1]
                elif i < n-1:
                    nums[i] = nums[i+1]
                else:
                    nums[i] = x

        # Calculate the maximum difference after replacement
        max_diff = 0
        for i in range(1, n):
            max_diff = max(max_diff, abs(nums[i] - nums[i-1]))

        return max_diff