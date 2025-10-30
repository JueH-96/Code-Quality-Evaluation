from typing import List
from collections import deque

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Convert nums to a deque for efficient pop and append operations
        nums = deque(nums)

        # Initialize the number of operations
        operations = 0

        # While the target is greater than 0
        while target > 0:
            # If nums is empty and target is still greater than 0, return -1
            if not nums:
                return -1

            # Get the maximum element from nums
            max_num = max(nums)

            # If the maximum element is greater than the target, return -1
            if max_num > target:
                return -1

            # If the maximum element is 1, it means we can't split it further
            if max_num == 1:
                # If the sum of all 1s is less than the target, return -1
                if nums.count(1) < target:
                    return -1
                else:
                    return operations

            # Remove the maximum element from nums
            nums.remove(max_num)

            # Add two occurrences of max_num / 2 to nums
            nums.append(max_num // 2)
            nums.append(max_num // 2)

            # Increment the number of operations
            operations += 1

            # Check if there is a subsequence that sums to the target
            current_sum = 0
            temp_nums = list(nums)
            for num in temp_nums:
                if current_sum + num <= target:
                    current_sum += num
                if current_sum == target:
                    return operations

        return operations