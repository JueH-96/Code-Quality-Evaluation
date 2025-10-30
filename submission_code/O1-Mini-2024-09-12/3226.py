class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        nums_sorted = sorted(nums)
        arr = []
        for i in range(0, len(nums_sorted), 2):
            alice = nums_sorted[i]
            bob = nums_sorted[i+1]
            arr.append(bob)
            arr.append(alice)
        return arr