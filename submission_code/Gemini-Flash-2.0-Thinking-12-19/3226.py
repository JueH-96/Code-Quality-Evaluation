class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        while nums:
            alice_removed = min(nums)
            nums.remove(alice_removed)
            bob_removed = min(nums)
            nums.remove(bob_removed)
            arr.append(bob_removed)
            arr.append(alice_removed)
        return arr