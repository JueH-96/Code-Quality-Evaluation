class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        expected_sum = n * n * (n * n + 1) // 2
        actual_sum = sum(sum(row) for row in grid)
        repeated = actual_sum - expected_sum

        flat_list = [num for row in grid for num in row]
        missing = (set(range(1, n * n + 1)) - set(flat_list)).pop()

        return [repeated, missing]