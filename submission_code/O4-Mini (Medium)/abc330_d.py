import sys
import threading

def main():
    data = sys.stdin
    line = data.readline()
    if not line:
        return
    N = int(line)
    grid = [data.readline().rstrip() for _ in range(N)]
    
    # Count 'o's in each row and column
    row_count = [0] * N
    col_count = [0] * N
    for i in range(N):
        row = grid[i]
        cnt = row.count('o')
        row_count[i] = cnt
        # For columns, we have to scan
        # but to save time we scan per character:
        # we'll accumulate per cell
    # Instead, do a single pass to fill col_count
    for i in range(N):
        row = grid[i]
        for j, ch in enumerate(row):
            if ch == 'o':
                col_count[j] += 1

    # Now sum over each 'o' cell: (row_count[i]-1)*(col_count[j]-1)
    total = 0
    for i in range(N):
        rc = row_count[i] - 1
        if rc <= 0:
            continue
        row = grid[i]
        for j, ch in enumerate(row):
            if ch == 'o':
                cc = col_count[j] - 1
                if cc > 0:
                    total += rc * cc

    print(total)

if __name__ == "__main__":
    main()