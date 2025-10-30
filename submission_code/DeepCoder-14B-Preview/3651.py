class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        for i in range(n):
            if nums[i] == 0:
                result[i] = 0
            else:
                if nums[i] > 0:
                    steps = nums[i]
                    new_index = (i + steps) % n
                else:
                    steps = -nums[i]
                    new_index = (i - steps) % n
                result[i] = nums[new_index]
        return result