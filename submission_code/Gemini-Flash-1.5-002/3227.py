class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        count = {}
        for row in grid:
            for num in row:
                if num in count:
                    count[num] += 1
                else:
                    count[num] = 1
        repeated = 0
        missing = 0
        for i in range(1, n * n + 1):
            if i in count:
                if count[i] == 2:
                    repeated = i
            else:
                missing = i
        return [repeated, missing]