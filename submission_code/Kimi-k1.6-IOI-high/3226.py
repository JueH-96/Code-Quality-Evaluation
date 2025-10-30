class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        nums_sorted = sorted(nums)
        for i in range(0, len(nums_sorted), 2):
            nums_sorted[i], nums_sorted[i+1] = nums_sorted[i+1], nums_sorted[i]
        return nums_sorted