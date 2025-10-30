class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        while nums:
            min_index_alice = nums.index(min(nums))
            alice_removed = nums.pop(min_index_alice)
            min_index_bob = nums.index(min(nums))
            bob_removed = nums.pop(min_index_bob)
            arr.append(bob_removed)
            arr.append(alice_removed)
        return arr