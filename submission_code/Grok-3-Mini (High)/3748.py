class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        for k in range(-(n - 1), n):  # k from -(n-1) to n-1 inclusive
            start_i = max(0, k)
            end_i = min(n - 1, n - 1 + k)
            if start_i <= end_i:
                # Collect values along the diagonal
                values = []
                for i in range(start_i, end_i + 1):
                    j = i - k
                    values.append(grid[i][j])
                
                # Sort the values based on k
                if k >= 0:
                    # Non-increasing order
                    values.sort(reverse=True)
                else:
                    # Non-decreasing order
                    values.sort()
                
                # Assign sorted values back to the diagonal
                idx = 0
                for i in range(start_i, end_i + 1):
                    j = i - k
                    grid[i][j] = values[idx]
                    idx += 1
        
        return grid