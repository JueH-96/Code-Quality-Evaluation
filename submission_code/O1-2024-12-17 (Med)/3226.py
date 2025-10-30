class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        while nums:
            # Alice removes the minimum element
            alice_val = min(nums)
            nums.remove(alice_val)
            # Bob removes the minimum element
            bob_val = min(nums)
            nums.remove(bob_val)
            # Bob's removed element goes first, then Alice's
            arr.append(bob_val)
            arr.append(alice_val)
        return arr