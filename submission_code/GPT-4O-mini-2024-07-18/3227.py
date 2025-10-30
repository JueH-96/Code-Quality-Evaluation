class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        expected_sum = n * n * (n * n + 1) // 2
        actual_sum = 0
        actual_sum_of_squares = 0
        
        seen = set()
        repeated = 0
        
        for row in grid:
            for num in row:
                actual_sum += num
                actual_sum_of_squares += num * num
                if num in seen:
                    repeated = num
                else:
                    seen.add(num)
        
        missing = expected_sum - (actual_sum - repeated)
        
        return [repeated, missing]