from typing import List

class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        while nums:
            # Alice removes the minimum element first.
            alice = min(nums)
            nums.remove(alice)
            # Bob removes the new minimum element.
            bob = min(nums)
            nums.remove(bob)
            # Bob appends his element first, then Alice.
            arr.append(bob)
            arr.append(alice)
        return arr