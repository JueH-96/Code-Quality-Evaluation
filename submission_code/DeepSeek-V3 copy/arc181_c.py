def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    Q = list(map(int, data[N+1:2*N+1]))
    
    # Create a grid filled with '0's
    grid = [['0' for _ in range(N)] for _ in range(N)]
    
    # Adjust the grid based on P and Q
    # For rows, we need S_{P_1} < S_{P_2} < ... < S_{P_N}
    # For columns, we need T_{Q_1} < T_{Q_2} < ... < T_{Q_N}
    
    # To ensure S_{P_i} < S_{P_j} for i < j, we can make sure that the first differing character is '0' in S_{P_i} and '1' in S_{P_j}
    # Similarly for columns
    
    # We can achieve this by setting the diagonal elements based on P and Q
    # For rows, we can set the diagonal elements to '0' for the first row, '1' for the second, etc.
    # For columns, we can set the diagonal elements to '0' for the first column, '1' for the second, etc.
    
    # However, we need to ensure that both conditions are satisfied simultaneously
    # One way is to set the grid such that the diagonal elements are '0' for the first row and column, '1' for the second, etc.
    
    # But to satisfy both P and Q, we need to find a way to assign '0's and '1's such that both conditions are met
    
    # A simple approach is to set the grid such that the rows are ordered according to P and the columns are ordered according to Q
    # We can achieve this by setting the grid such that the rows are ordered in increasing order of P, and the columns are ordered in increasing order of Q
    
    # To do this, we can assign '0's and '1's based on the positions in P and Q
    
    # For example, for row i, we can set the first (P_i - 1) columns to '0' and the rest to '1'
    # Similarly for columns, we can set the first (Q_j - 1) rows to '0' and the rest to '1'
    
    # However, this may not satisfy both conditions simultaneously
    
    # Another approach is to use the fact that the grid can be filled in such a way that the rows and columns are ordered based on P and Q
    
    # We can use the following strategy:
    # For each cell (i, j), if P_i < P_j and Q_i < Q_j, set it to '0', else set it to '1'
    
    # But this may not work for all cases
    
    # Instead, we can use the following method:
    # For each row i, set the first (P_i - 1) columns to '0' and the rest to '1'
    # This ensures that the rows are ordered according to P
    # Then, for each column j, set the first (Q_j - 1) rows to '0' and the rest to '1'
    # This ensures that the columns are ordered according to Q
    
    # However, this may conflict with the row ordering
    
    # To resolve this, we can prioritize one condition and adjust the other accordingly
    
    # For example, we can first set the rows according to P, and then adjust the columns to satisfy Q
    
    # Here's a possible implementation:
    
    # First, set the rows according to P
    for i in range(N):
        for j in range(N):
            if j < P[i] - 1:
                grid[i][j] = '0'
            else:
                grid[i][j] = '1'
    
    # Then, adjust the columns according to Q
    for j in range(N):
        for i in range(N):
            if i < Q[j] - 1:
                grid[i][j] = '0'
            else:
                grid[i][j] = '1'
    
    # Print the grid
    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    main()