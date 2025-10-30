class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        result = []
        for i in range(n - k + 1):
            if all(nums[i + j + 1] - nums[i + j] == 1 for j in range(k - 1)):
                result.append(nums[i + k - 1])
            else:
                result.append(-1)
        return result