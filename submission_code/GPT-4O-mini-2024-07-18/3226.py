from typing import List

class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        nums.sort()  # Sort the array to easily get the minimum elements
        
        while nums:
            # Alice removes the minimum element
            alice_pick = nums.pop(0)
            # Bob removes the next minimum element
            bob_pick = nums.pop(0)
            
            # Bob appends his pick first
            arr.append(bob_pick)
            # Alice appends her pick
            arr.append(alice_pick)
        
        return arr