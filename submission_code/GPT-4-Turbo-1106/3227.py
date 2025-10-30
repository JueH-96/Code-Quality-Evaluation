class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        expected_sum = n * n * (n * n + 1) // 2
        actual_sum = sum(sum(row) for row in grid)
        flat_grid = [item for sublist in grid for item in sublist]
        actual_sum_of_squares = sum(x*x for x in flat_grid)
        expected_sum_of_squares = sum(i*i for i in range(1, n*n + 1))
        
        # Calculate the difference between the expected and actual sums
        sum_diff = expected_sum - actual_sum
        square_sum_diff = expected_sum_of_squares - actual_sum_of_squares
        
        # Calculate the missing and repeated numbers
        # sum_diff = b - a
        # square_sum_diff = b^2 - a^2 = (b - a)(b + a)
        # (b + a) = square_sum_diff / sum_diff
        b_plus_a = square_sum_diff // sum_diff
        b = (b_plus_a + sum_diff) // 2
        a = b - sum_diff
        
        return [a, b]