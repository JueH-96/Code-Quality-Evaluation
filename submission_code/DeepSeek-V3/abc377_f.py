# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    rows = set()
    cols = set()
    diag1 = set()
    diag2 = set()
    
    index = 2
    for _ in range(M):
        a = int(data[index])
        b = int(data[index+1])
        rows.add(a)
        cols.add(b)
        diag1.add(a + b)
        diag2.add(a - b)
        index += 2
    
    total = N * N
    occupied = M
    unsafe_rows = len(rows)
    unsafe_cols = len(cols)
    unsafe_diag1 = len(diag1)
    unsafe_diag2 = len(diag2)
    
    # Calculate the number of squares that are unsafe due to rows, columns, and diagonals
    # Using inclusion-exclusion principle
    # Total unsafe squares = (unsafe_rows * N) + (unsafe_cols * N) - (unsafe_rows * unsafe_cols)
    # Plus the squares on the diagonals, but need to subtract overlaps
    
    # First, calculate the number of squares that are in unsafe rows or columns
    unsafe_row_col = unsafe_rows * N + unsafe_cols * N - unsafe_rows * unsafe_cols
    
    # Now, calculate the number of squares that are on unsafe diagonals
    # For each diagonal in diag1, the number of squares is min(N, a + b - 1) - max(1, a + b - N) + 1
    # Similarly for diag2
    
    # To simplify, since N can be up to 1e9, we need a smarter way
    # For diag1: a + b = k, where k ranges from 2 to 2N
    # The number of squares on diag1 is min(k-1, N) - max(1, k - N) + 1
    # Similarly for diag2: a - b = k, where k ranges from -(N-1) to (N-1)
    # The number of squares on diag2 is min(N - k, N) - max(1, 1 - k) + 1
    
    # But since we have the set of diag1 and diag2, we can calculate the total number of squares on these diagonals
    # For each k in diag1, the number of squares is min(k-1, N) - max(1, k - N) + 1
    # For each k in diag2, the number of squares is min(N - k, N) - max(1, 1 - k) + 1
    
    # Initialize the total unsafe squares on diagonals
    unsafe_diag_total = 0
    
    for k in diag1:
        low = max(1, k - N)
        high = min(k - 1, N)
        count = high - low + 1
        unsafe_diag_total += count
    
    for k in diag2:
        low = max(1, 1 - k)
        high = min(N - k, N)
        count = high - low + 1
        unsafe_diag_total += count
    
    # Now, we need to subtract the overlaps between rows/columns and diagonals
    # For each piece, it is counted in both rows/columns and diagonals
    # So, the total unsafe squares is unsafe_row_col + unsafe_diag_total - M
    
    total_unsafe = unsafe_row_col + unsafe_diag_total - M
    
    # The number of safe squares is total - total_unsafe
    safe_squares = total - total_unsafe
    
    print(safe_squares)

if __name__ == "__main__":
    main()