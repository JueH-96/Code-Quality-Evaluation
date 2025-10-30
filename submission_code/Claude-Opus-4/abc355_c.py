# YOUR CODE HERE
def solve():
    N, T = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Track marked cells
    marked = [[False] * N for _ in range(N)]
    
    # Count marked cells in each row, column, and diagonal
    row_count = [0] * N
    col_count = [0] * N
    diag1_count = 0  # top-left to bottom-right
    diag2_count = 0  # top-right to bottom-left
    
    for turn in range(T):
        value = A[turn]
        # Convert value to row and column (0-indexed)
        row = (value - 1) // N
        col = (value - 1) % N
        
        # Mark the cell
        if not marked[row][col]:
            marked[row][col] = True
            
            # Update counts
            row_count[row] += 1
            col_count[col] += 1
            
            # Check if on main diagonal
            if row == col:
                diag1_count += 1
            
            # Check if on anti-diagonal
            if row + col == N - 1:
                diag2_count += 1
            
            # Check for bingo
            if (row_count[row] == N or 
                col_count[col] == N or 
                diag1_count == N or 
                diag2_count == N):
                print(turn + 1)
                return
    
    print(-1)

solve()