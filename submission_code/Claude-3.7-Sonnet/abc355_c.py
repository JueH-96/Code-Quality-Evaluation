def main():
    N, T = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Create counters for rows and columns
    row_count = [0] * N
    col_count = [0] * N
    
    # Create a grid to track which cells have been marked
    grid = [[False for _ in range(N)] for _ in range(N)]
    
    for t in range(T):
        num = A[t]
        
        # Convert the number to row and column indices
        row = (num - 1) // N
        col = (num - 1) % N
        
        # Mark the cell
        grid[row][col] = True
        row_count[row] += 1
        col_count[col] += 1
        
        # Check if the row or column is completely marked
        if row_count[row] == N or col_count[col] == N:
            return t + 1
        
        # Check if the top-left to bottom-right diagonal is completely marked
        diag1_complete = True
        for i in range(N):
            if not grid[i][i]:
                diag1_complete = False
                break
        
        # Check if the top-right to bottom-left diagonal is completely marked
        diag2_complete = True
        for i in range(N):
            if not grid[i][N-1-i]:
                diag2_complete = False
                break
        
        if diag1_complete or diag2_complete:
            return t + 1
    
    return -1

result = main()
print(result)