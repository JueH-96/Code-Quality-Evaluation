class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        while nums:
            # Alice removes the minimum element
            alice_remove = min(nums)
            nums.remove(alice_remove)

            # Bob removes the minimum element
            bob_remove = min(nums)
            nums.remove(bob_remove)

            # Bob appends his removed element first
            arr.append(bob_remove)

            # Alice appends her removed element
            arr.append(alice_remove)

        return arr