class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        result = []
        for i in range(n - k + 1):
            valid = True
            for j in range(i + 1, i + k):
                if nums[j] - nums[j - 1] != 1:
                    valid = False
                    break
            if valid:
                result.append(nums[i + k - 1])
            else:
                result.append(-1)
        return result