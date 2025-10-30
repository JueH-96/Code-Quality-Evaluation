class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        sorted_nums = sorted(nums)
        arr = []
        for i in range(0, len(sorted_nums), 2):
            arr.append(sorted_nums[i + 1])
            arr.append(sorted_nums[i])
        return arr