class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        while nums:
            # Alice removes the minimum element
            alice_min = min(nums)
            nums.remove(alice_min)
            # Bob removes the next minimum element
            bob_min = min(nums)
            nums.remove(bob_min)
            # Bob appends his element first
            arr.append(bob_min)
            # Alice appends her element next
            arr.append(alice_min)
        return arr