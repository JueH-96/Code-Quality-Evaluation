from bisect import bisect_left
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx]); idx +=1
    W = int(data[idx]); idx +=1
    M = int(data[idx]); idx +=1

    row_last = [ (0, 0) ] * (H + 1)  # 1-based indexing for rows
    col_last = [ (0, 0) ] * (W + 1)  # 1-based indexing for columns

    for step in range(1, M + 1):
        T = int(data[idx]); idx +=1
        A = int(data[idx]); idx +=1
        X = int(data[idx]); idx +=1
        if T == 1:
            row_last[A] = (step, X)
        else:
            col_last[A] = (step, X)

    # Collect all column steps and row steps
    col_steps = [ col_last[c][0] for c in range(1, W+1) ]
    row_steps = [ row_last[r][0] for r in range(1, H+1) ]

    # Sort these steps for binary search
    sorted_col_steps = sorted(col_steps)
    sorted_row_steps = sorted(row_steps)

    color_counts = defaultdict(int)

    # Process each row
    for r in range(1, H + 1):
        step_r, color_r = row_last[r]
        if step_r == 0:
            continue
        # Number of columns with step < step_r
        count = bisect_left(sorted_col_steps, step_r)
        color_counts[color_r] += count

    # Process each column
    for c in range(1, W + 1):
        step_c, color_c = col_last[c]
        if step_c == 0:
            continue
        # Number of rows with step < step_c
        count = bisect_left(sorted_row_steps, step_c)
        color_counts[color_c] += count

    # Calculate the remaining cells for color 0
    sum_painted = sum(color_counts.values())
    color_0 = H * W - sum_painted
    if color_0 > 0:
        color_counts[0] += color_0

    # Collect the results, excluding colors with zero count
    result = []
    for color in color_counts:
        if color_counts[color] > 0:
            result.append( (color, color_counts[color]) )

    # Sort the result by color
    result.sort()

    # Output
    print(len(result))
    for color, count in result:
        print(color, count)

if __name__ == "__main__":
    main()