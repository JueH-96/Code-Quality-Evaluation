class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        nums = [0] * (n*n + 1)
        for row in grid:
            for num in row:
                nums[num] += 1
        ans = []
        for i in range(1, n*n + 1):
            if nums[i] == 2:
                ans.append(i)
            elif nums[i] == 0:
                ans.append(i)
        return ans