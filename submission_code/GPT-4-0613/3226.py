class Solution:
    def numberGame(self, nums):
        nums.sort()
        arr = []
        while nums:
            alice = nums.pop(nums.index(min(nums)))
            bob = nums.pop(nums.index(min(nums)))
            arr.append(bob)
            arr.append(alice)
        return arr