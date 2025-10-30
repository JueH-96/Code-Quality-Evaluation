from typing import List

class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        # Sort the input array in non-decreasing order.
        # This makes it easy to find the minimum and second minimum elements
        # in each conceptual round of the game.
        # After sorting, the elements are nums[0], nums[1], nums[2], ...
        # In the first round, Alice removes nums[0], Bob removes nums[1].
        # In the second round, Alice removes nums[2], Bob removes nums[3], and so on.
        nums.sort()

        # Initialize the result array where elements will be stored
        # in the order they are appended by Bob and Alice.
        arr = []

        # Iterate through the sorted array in steps of 2.
        # Each step covers one round of the game where two elements are removed
        # from nums and appended to arr.
        for i in range(0, len(nums), 2):
            # In the current step (round), nums[i] is the smallest remaining element,
            # and nums[i+1] is the second smallest remaining element.
            # Alice removes the minimum, which is nums[i].
            alice_pick = nums[i]
            # Bob removes the minimum from the remaining elements, which is nums[i+1].
            bob_pick = nums[i+1]

            # According to the rules, Bob appends his removed element first to arr.
            arr.append(bob_pick)
            # Then, Alice appends her removed element to arr.
            arr.append(alice_pick)

        # The loop finishes when all elements from the original nums
        # have been processed in pairs, meaning nums is conceptually empty.
        # Return the final resulting array arr.
        return arr