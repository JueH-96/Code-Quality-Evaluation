class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        while nums:
            # Alice's move
            alice_min = min(nums)
            nums.remove(alice_min)

            # Bob's move
            bob_min = min(nums)
            nums.remove(bob_min)

            # Append to arr
            arr.append(bob_min)
            arr.append(alice_min)

        return arr