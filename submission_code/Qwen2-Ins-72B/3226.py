class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        nums.sort()
        result = []
        for i in range(len(nums) - 1, -1, -2):
            result.append(nums[i])
            if i - 1 >= 0:
                result.append(nums[i - 1])
        return result