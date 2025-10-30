from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        valid_count = 0

        def is_valid(start, direction):
            curr = start
            while 0 <= curr < n:
                if nums[curr] == 0:
                    curr += direction
                else:
                    nums[curr] -= 1
                    direction = -direction
                    curr += direction
            return all(x == 0 for x in nums)

        for i in range(n):
            if nums[i] == 0:
                nums_copy = nums.copy()
                if is_valid(i, 1):  # Try moving right
                    valid_count += 1
                nums_copy = nums.copy()
                if is_valid(i, -1):  # Try moving left
                    valid_count += 1

        return valid_count