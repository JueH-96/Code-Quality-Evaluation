from typing import List

class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        while nums:
            # Alice's move
            alice_num = min(nums)
            nums.remove(alice_num)
            
            # Bob's move
            bob_num = min(nums)
            nums.remove(bob_num)
            
            # Append to arr
            arr.append(bob_num)
            arr.append(alice_num)
        return arr