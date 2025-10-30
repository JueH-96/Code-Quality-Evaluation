class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        counts = {}
        for r in range(n):
            for c in range(n):
                num = grid[r][c]
                counts[num] = counts.get(num, 0) + 1

        repeated = -1
        missing = -1
        expected_nums = set(range(1, n * n + 1))
        grid_nums = set()
        for r in range(n):
            for c in range(n):
                grid_nums.add(grid[r][c])

        for num in counts:
            if counts[num] == 2:
                repeated = num
                break

        for i in range(1, n * n + 1):
            if i not in grid_nums:
                missing = i
                break
        
        if missing == -1:
            all_present = set(range(1, n * n + 1))
            grid_set = set()
            for r in range(n):
                for c in range(n):
                    grid_set.add(grid[r][c])
            missing_set = all_present - grid_set
            missing = list(missing_set)[0]


        return [repeated, missing]