class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        expected_sum = n * n * (n * n + 1) // 2
        actual_sum = sum(sum(row) for row in grid)
        
        number_set = set()
        repeated = 0
        
        for row in grid:
            for num in row:
                if num in number_set:
                    repeated = num
                else:
                    number_set.add(num)
        
        missing = expected_sum - actual_sum + repeated
        
        return [repeated, missing]