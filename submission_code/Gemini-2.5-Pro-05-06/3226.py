from typing import List

class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        
        # Step 1: Sort the input array `nums` in non-decreasing order.
        # This allows us to easily determine Alice's and Bob's choices in each round
        # by simply taking consecutive elements from the sorted list.
        nums.sort()
        
        # Step 2: Initialize an empty list `arr`. This list will store the elements
        # in the order specified by the game rules and will be the final result.
        arr = []
        
        # Step 3: Iterate through the sorted `nums` array.
        # We process elements in pairs, so the loop increments `i` by 2.
        # The problem guarantees that `nums.length` is even, so `nums[i+1]` will always be valid.
        for i in range(0, len(nums), 2):
            # In the conceptual round based on the sorted list:
            # Alice removes `nums[i]` (the smallest element currently available).
            alice_removed_element = nums[i]
            # Bob removes `nums[i+1]` (the smallest element from the remaining ones,
            # which is the next element in the sorted list).
            bob_removed_element = nums[i+1]
            
            # Step 4: Append elements to `arr` according to the game rules.
            # First, Bob appends his removed element.
            arr.append(bob_removed_element)
            # Then, Alice appends her removed element.
            arr.append(alice_removed_element)
            
        # Step 5: Return the resulting array `arr`.
        return arr