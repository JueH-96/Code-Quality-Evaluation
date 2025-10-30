import collections
from typing import List

class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        # Step 1: Sort the input array nums in ascending order.
        # This is crucial because Alice and Bob always remove the minimum
        # elements available. If the array is sorted, the elements Alice takes
        # will be nums[0], nums[2], nums[4], etc., and the elements Bob takes
        # will be nums[1], nums[3], nums[5], etc.
        nums.sort()

        # Step 2: Initialize an empty list to store the results.
        arr = []

        # Step 3: Iterate through the sorted nums array in steps of 2.
        # Since nums has an even length, we can always process elements in pairs.
        # 'i' will represent the index of Alice's element in the current pair.
        for i in range(0, len(nums), 2):
            # In each round:
            # Alice removes nums[i] (the current minimum)
            # Bob removes nums[i+1] (the minimum from the remaining elements)

            # Rule: Bob appends his element first.
            arr.append(nums[i+1])

            # Rule: Then Alice appends her element.
            arr.append(nums[i])

        # Step 4: Return the resulting array.
        return arr