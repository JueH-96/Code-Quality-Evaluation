class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        nums.sort()
        arr = []
        while nums:
            alice_min = nums.pop(0)
            bob_min = nums.pop(0)
            arr.append(bob_min)
            arr.append(alice_min)
        return arr