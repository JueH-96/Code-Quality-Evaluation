from typing import List

class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        # Sort the array so we know the removal order of minima
        nums.sort()
        arr = []
        # Each round we take two elements: the current smallest (Alice) and next smallest (Bob)
        # Bob appends his removed element first, then Alice appends hers
        for i in range(0, len(nums), 2):
            # nums[i] is Alice's removal, nums[i+1] is Bob's removal
            arr.append(nums[i+1])  # Bob appends first
            arr.append(nums[i])    # Alice appends second
        return arr