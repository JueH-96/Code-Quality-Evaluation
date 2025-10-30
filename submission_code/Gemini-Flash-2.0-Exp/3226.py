class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        nums.sort()
        while nums:
            alice = nums.pop(0)
            bob = nums.pop(0)
            arr.append(bob)
            arr.append(alice)
        return arr