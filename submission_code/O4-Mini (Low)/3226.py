from typing import List

class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        # Sort the list to simulate repeatedly removing the minimum elements
        nums.sort()
        res = []
        # Process pairs: in each round Alice removes nums[i] (min), Bob removes nums[i+1] (next min),
        # but they append to arr in order: Bob's removal first, then Alice's.
        for i in range(0, len(nums), 2):
            # nums[i] is Alice's removal, nums[i+1] is Bob's removal
            res.append(nums[i+1])  # Bob appends his removed element
            res.append(nums[i])    # Alice appends her removed element
        return res