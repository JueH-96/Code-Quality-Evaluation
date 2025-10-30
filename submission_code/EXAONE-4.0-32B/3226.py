class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        sorted_nums = sorted(nums)
        result = []
        n = len(nums)
        for i in range(0, n, 2):
            result.append(sorted_nums[i + 1])
            result.append(sorted_nums[i])
        return result