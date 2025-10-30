import sys
import bisect

def main():
    H, W, Q = map(int, sys.stdin.readline().split())
    # Initialize walls
    # For each row, maintain a list of columns that have walls
    # Similarly, for each column, maintain a list of rows that have walls
    # We can use sets for quick lookup and removal
    # But since we need to find the first wall in a direction, we need ordered data structures
    # So, for each row, we maintain a sorted list of columns with walls
    # Similarly, for each column, we maintain a sorted list of rows with walls
    # Initialize all cells as having walls
    # So, for each row, the list of columns is [1, 2, ..., W]
    # For each column, the list of rows is [1, 2, ..., H]
    # We can represent these as lists of lists
    row_walls = [sorted(range(1, W+1)) for _ in range(H+1)]  # row_walls[i] is the list of columns with walls in row i
    col_walls = [sorted(range(1, H+1)) for _ in range(W+1)]  # col_walls[j] is the list of rows with walls in column j
    
    # To handle the queries, we need to efficiently find and remove walls
    # For each query (R, C):
    # 1. Check if (R, C) has a wall
    # 2. If yes, remove it
    # 3. If no, find the first walls in the four directions and remove them
    # To find the first wall in a direction, we can use binary search on the sorted lists
    # For example, to find the first wall above (R, C), we look for the largest element in col_walls[C] that is less than R
    # Similarly for the other directions
    
    total_walls = H * W
    for _ in range(Q):
        R, C = map(int, sys.stdin.readline().split())
        # Check if (R, C) has a wall
        # To check if C is in row_walls[R], we can use bisect
        idx = bisect.bisect_left(row_walls[R], C)
        if idx < len(row_walls[R]) and row_walls[R][idx] == C:
            # Wall exists at (R, C), remove it
            row_walls[R].pop(idx)
            # Also remove R from col_walls[C]
            idx_col = bisect.bisect_left(col_walls[C], R)
            if idx_col < len(col_walls[C]) and col_walls[C][idx_col] == R:
                col_walls[C].pop(idx_col)
            total_walls -= 1
        else:
            # No wall at (R, C), find the first walls in the four directions
            # Up: find the largest row less than R in col_walls[C]
            idx_up = bisect.bisect_left(col_walls[C], R) - 1
            if idx_up >= 0:
                row_up = col_walls[C][idx_up]
                # Remove (row_up, C)
                idx_row = bisect.bisect_left(row_walls[row_up], C)
                if idx_row < len(row_walls[row_up]) and row_walls[row_up][idx_row] == C:
                    row_walls[row_up].pop(idx_row)
                    col_walls[C].pop(idx_up)
                    total_walls -= 1
            # Down: find the smallest row greater than R in col_walls[C]
            idx_down = bisect.bisect_right(col_walls[C], R)
            if idx_down < len(col_walls[C]):
                row_down = col_walls[C][idx_down]
                # Remove (row_down, C)
                idx_row = bisect.bisect_left(row_walls[row_down], C)
                if idx_row < len(row_walls[row_down]) and row_walls[row_down][idx_row] == C:
                    row_walls[row_down].pop(idx_row)
                    col_walls[C].pop(idx_down)
                    total_walls -= 1
            # Left: find the largest column less than C in row_walls[R]
            idx_left = bisect.bisect_left(row_walls[R], C) - 1
            if idx_left >= 0:
                col_left = row_walls[R][idx_left]
                # Remove (R, col_left)
                idx_col = bisect.bisect_left(col_walls[col_left], R)
                if idx_col < len(col_walls[col_left]) and col_walls[col_left][idx_col] == R:
                    col_walls[col_left].pop(idx_col)
                    row_walls[R].pop(idx_left)
                    total_walls -= 1
            # Right: find the smallest column greater than C in row_walls[R]
            idx_right = bisect.bisect_right(row_walls[R], C)
            if idx_right < len(row_walls[R]):
                col_right = row_walls[R][idx_right]
                # Remove (R, col_right)
                idx_col = bisect.bisect_left(col_walls[col_right], R)
                if idx_col < len(col_walls[col_right]) and col_walls[col_right][idx_col] == R:
                    col_walls[col_right].pop(idx_col)
                    row_walls[R].pop(idx_right)
                    total_walls -= 1
    print(total_walls)

if __name__ == "__main__":
    main()