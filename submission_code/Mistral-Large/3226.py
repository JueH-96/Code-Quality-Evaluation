from typing import List

class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        # Sort the array to easily access the minimum elements
        nums.sort()

        # Initialize the result array
        arr = []

        # Process the elements in pairs
        for i in range(0, len(nums), 2):
            # Bob appends the second minimum element first
            arr.append(nums[i + 1])
            # Alice appends the minimum element second
            arr.append(nums[i])

        return arr