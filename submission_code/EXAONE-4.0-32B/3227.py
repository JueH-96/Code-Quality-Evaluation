class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        n_sq = n * n
        S = n_sq * (n_sq + 1) // 2
        T = n_sq * (n_sq + 1) * (2 * n_sq + 1) // 6
        
        total_sum = 0
        total_sq = 0
        for row in grid:
            for num in row:
                total_sum += num
                total_sq += num * num
        
        D = total_sum - S
        D_sq = total_sq - T
        a_plus_b = D_sq // D
        
        a = (D + a_plus_b) // 2
        b = (a_plus_b - D) // 2
        
        return [a, b]