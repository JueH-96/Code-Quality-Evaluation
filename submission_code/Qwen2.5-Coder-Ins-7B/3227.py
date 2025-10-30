class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        total_sum = n * (n + 1) // 2
        total_square_sum = n * (n + 1) * (2 * n + 1) // 6
        actual_sum = sum(sum(row) for row in grid)
        actual_square_sum = sum(sum(x * x for x in row) for row in grid)
        
        diff = actual_sum - total_sum
        diff_square = actual_square_sum - total_square_sum
        
        a = (diff_square // diff + diff) // 2
        b = a - diff
        
        return [a, b]