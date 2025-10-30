class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        res = []
        for i in range(n - k + 1):
            valid = True
            for j in range(i, i + k - 1):
                if nums[j] >= nums[j + 1]:
                    valid = False
                    break
            if valid and (nums[i + k - 1] - nums[i] == k - 1):
                res.append(nums[i + k - 1])
            else:
                res.append(-1)
        return res