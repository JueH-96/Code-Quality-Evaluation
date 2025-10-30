class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        nums = [0] * (n * n + 1)
        duplicate = -1
        missing = -1

        for row in grid:
            for num in row:
                if nums[num] == 0:
                    nums[num] = 1
                else:
                    duplicate = num

        for i in range(1, len(nums)):
            if nums[i] == 0:
                missing = i
                break

        return [duplicate, missing]