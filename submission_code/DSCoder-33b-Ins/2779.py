class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        res = []
        count = 0
        for i, j in queries:
            if i > 0 and nums[i] == nums[i - 1] and nums[i] != 0:
                count -= 1
            if i < n - 1 and nums[i] == nums[i + 1] and nums[i] != 0:
                count -= 1
            nums[i] = j
            if i > 0 and nums[i] == nums[i - 1] and nums[i] != 0:
                count += 1
            if i < n - 1 and nums[i] == nums[i + 1] and nums[i] != 0:
                count += 1
            res.append(count)
        return res