class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        len_n = len(nums)
        result = []
        for i in range(len_n):
            s = nums[i]
            if s == 0:
                result.append(0)
            else:
                new_index = (i + s) % len_n
                result.append(nums[new_index])
        return result