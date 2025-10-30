class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        while nums:
            # Alice removes the minimum
            alice_min = min(nums)
            nums.remove(alice_min)
            # Bob removes the next minimum
            bob_min = min(nums)
            nums.remove(bob_min)
            # Bob appends first, then Alice
            arr.append(bob_min)
            arr.append(alice_min)
        return arr