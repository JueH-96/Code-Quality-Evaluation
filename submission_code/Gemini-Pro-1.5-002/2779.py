class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        ans = []
        count = 0
        for index, color in queries:
            if index > 0 and nums[index - 1] == nums[index] and nums[index] != 0:
                count -= 1
            if index < n - 1 and nums[index + 1] == nums[index] and nums[index] != 0:
                count -= 1
            nums[index] = color
            if index > 0 and nums[index - 1] == nums[index] and nums[index] != 0:
                count += 1
            if index < n - 1 and nums[index + 1] == nums[index] and nums[index] != 0:
                count += 1
            ans.append(count)
        return ans