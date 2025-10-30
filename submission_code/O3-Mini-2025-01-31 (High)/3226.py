from typing import List

class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        while nums:
            # Alice removes the minimum element from nums
            alice_elem = min(nums)
            nums.remove(alice_elem)
            # Bob removes the next minimum element from nums
            bob_elem = min(nums)
            nums.remove(bob_elem)
            # Bob appends his removed element first, then Alice appends hers
            arr.append(bob_elem)
            arr.append(alice_elem)
        return arr