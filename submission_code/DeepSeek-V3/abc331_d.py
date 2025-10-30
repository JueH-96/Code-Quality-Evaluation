# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    P = []
    for _ in range(N):
        P.append(data[idx])
        idx += 1
    # Precompute the prefix sums for the pattern
    prefix = [[0]*(N+1) for _ in range(N+1)]
    for i in range(N):
        row_sum = 0
        for j in range(N):
            if P[i][j] == 'B':
                row_sum += 1
            prefix[i+1][j+1] = prefix[i][j+1] + row_sum
    # Process each query
    for _ in range(Q):
        A = int(data[idx])
        B = int(data[idx+1])
        C = int(data[idx+2])
        D = int(data[idx+3])
        idx += 4
        # Calculate the number of full patterns in the rectangle
        # For rows: A to C
        # For columns: B to D
        # The pattern repeats every N rows and N columns
        # So, the full patterns are:
        # rows: (A // N) * N to (C // N) * N - 1
        # columns: (B // N) * N to (D // N) * N - 1
        # The remaining parts are the edges
        # Calculate the number of full patterns in rows and columns
        row_start = A // N
        row_end = C // N
        col_start = B // N
        col_end = D // N
        # Full patterns count
        full_row = row_end - row_start - 1
        full_col = col_end - col_start - 1
        if full_row < 0:
            full_row = 0
        if full_col < 0:
            full_col = 0
        # Total full patterns
        total_full = full_row * full_col
        # Each full pattern has prefix[N][N] black cells
        black_full = total_full * prefix[N][N]
        # Now, handle the edges
        # Left edge: columns from B to (col_start + 1) * N - 1
        # Right edge: columns from col_end * N to D
        # Top edge: rows from A to (row_start + 1) * N - 1
        # Bottom edge: rows from row_end * N to C
        # Calculate the black cells in the left edge
        left_col_start = B
        left_col_end = (col_start + 1) * N - 1
        if left_col_end < left_col_start:
            left_col_end = left_col_start - 1
        # Calculate the black cells in the right edge
        right_col_start = col_end * N
        right_col_end = D
        if right_col_end < right_col_start:
            right_col_end = right_col_start - 1
        # Calculate the black cells in the top edge
        top_row_start = A
        top_row_end = (row_start + 1) * N - 1
        if top_row_end < top_row_start:
            top_row_end = top_row_start - 1
        # Calculate the black cells in the bottom edge
        bottom_row_start = row_end * N
        bottom_row_end = C
        if bottom_row_end < bottom_row_start:
            bottom_row_end = bottom_row_start - 1
        # Now, calculate the black cells in the edges
        # Left edge: rows from top_row_start to bottom_row_end, columns from left_col_start to left_col_end
        # Right edge: rows from top_row_start to bottom_row_end, columns from right_col_start to right_col_end
        # Top edge: rows from top_row_start to top_row_end, columns from left_col_start to right_col_end
        # Bottom edge: rows from bottom_row_start to bottom_row_end, columns from left_col_start to right_col_end
        # Calculate the black cells in the left edge
        left_black = 0
        if left_col_start <= left_col_end:
            # Rows: top_row_start to bottom_row_end
            # Columns: left_col_start to left_col_end
            # For each row, calculate the number of black cells in the columns
            for row in range(top_row_start, bottom_row_end + 1):
                row_mod = row % N
                col_start_mod = left_col_start % N
                col_end_mod = left_col_end % N
                if col_start_mod <= col_end_mod:
                    left_black += prefix[row_mod+1][col_end_mod+1] - prefix[row_mod+1][col_start_mod]
                else:
                    left_black += prefix[row_mod+1][N] - prefix[row_mod+1][col_start_mod] + prefix[row_mod+1][col_end_mod+1]
        # Calculate the black cells in the right edge
        right_black = 0
        if right_col_start <= right_col_end:
            # Rows: top_row_start to bottom_row_end
            # Columns: right_col_start to right_col_end
            for row in range(top_row_start, bottom_row_end + 1):
                row_mod = row % N
                col_start_mod = right_col_start % N
                col_end_mod = right_col_end % N
                if col_start_mod <= col_end_mod:
                    right_black += prefix[row_mod+1][col_end_mod+1] - prefix[row_mod+1][col_start_mod]
                else:
                    right_black += prefix[row_mod+1][N] - prefix[row_mod+1][col_start_mod] + prefix[row_mod+1][col_end_mod+1]
        # Calculate the black cells in the top edge
        top_black = 0
        if top_row_start <= top_row_end:
            # Rows: top_row_start to top_row_end
            # Columns: left_col_start to right_col_end
            for row in range(top_row_start, top_row_end + 1):
                row_mod = row % N
                col_start_mod = left_col_start % N
                col_end_mod = right_col_end % N
                if col_start_mod <= col_end_mod:
                    top_black += prefix[row_mod+1][col_end_mod+1] - prefix[row_mod+1][col_start_mod]
                else:
                    top_black += prefix[row_mod+1][N] - prefix[row_mod+1][col_start_mod] + prefix[row_mod+1][col_end_mod+1]
        # Calculate the black cells in the bottom edge
        bottom_black = 0
        if bottom_row_start <= bottom_row_end:
            # Rows: bottom_row_start to bottom_row_end
            # Columns: left_col_start to right_col_end
            for row in range(bottom_row_start, bottom_row_end + 1):
                row_mod = row % N
                col_start_mod = left_col_start % N
                col_end_mod = right_col_end % N
                if col_start_mod <= col_end_mod:
                    bottom_black += prefix[row_mod+1][col_end_mod+1] - prefix[row_mod+1][col_start_mod]
                else:
                    bottom_black += prefix[row_mod+1][N] - prefix[row_mod+1][col_start_mod] + prefix[row_mod+1][col_end_mod+1]
        # Total black cells is the sum of full patterns and edges
        total_black = black_full + left_black + right_black + top_black + bottom_black
        print(total_black)

if __name__ == "__main__":
    main()