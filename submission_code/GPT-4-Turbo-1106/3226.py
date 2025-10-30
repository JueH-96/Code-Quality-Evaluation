class Solution:
    def numberGame(self, nums: list[int]) -> list[int]:
        arr = []
        while nums:
            # Alice removes the minimum element
            alice_pick = min(nums)
            nums.remove(alice_pick)
            # Bob removes the minimum element
            bob_pick = min(nums)
            nums.remove(bob_pick)
            # Bob appends his pick first
            arr.append(bob_pick)
            # Alice appends her pick
            arr.append(alice_pick)
        return arr