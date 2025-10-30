import sys

def solve():
    H, W = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        line = sys.stdin.readline().strip()
        row = [int(c) for c in line]
        grid.append(row)
    
    min_total = float('inf')
    
    # We will try all possible column flips (mask represents which columns are flipped)
    for mask in range(0, 1 << W):
        total = 0
        # For each row, decide whether to flip it or not
        for i in range(H):
            row_sum = 0
            for j in range(W):
                original = grid[i][j]
                # Check if column j is flipped (mask's j-th bit is set)
                if mask & (1 << j):
                    flipped = 1 - original
                else:
                    flipped = original
                row_sum += flipped
            # Decide whether flipping the row is better
            flipped_row_sum = W - row_sum
            total += min(row_sum, flipped_row_sum)
        if total < min_total:
            min_total = total
    print(min_total)

solve()