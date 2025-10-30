from typing import List

class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        """
        Solves the number game by sorting the array and swapping adjacent elements.

        The game rules dictate that in each round, the two smallest elements are
        removed. Let's say they are `alice_num` and `bob_num`, where `alice_num < bob_num`.
        Alice takes `alice_num`, and Bob takes `bob_num`. They are then appended to the
        result array in the order `[bob_num, alice_num]`.

        This process is equivalent to sorting the initial array and then swapping
        every pair of adjacent elements.
        """
        
        # Sort the array in non-decreasing order.
        # This groups the numbers that will be picked in each round into adjacent pairs.
        nums.sort()
        
        # The result array.
        arr = []
        
        # Iterate through the sorted array, processing two elements at a time.
        # The length of nums is guaranteed to be even.
        for i in range(0, len(nums), 2):
            # In each pair (nums[i], nums[i+1]), nums[i] is Alice's number
            # and nums[i+1] is Bob's number.
            
            # Bob appends his number first.
            arr.append(nums[i+1])
            # Then Alice appends her number.
            arr.append(nums[i])
            
        return arr