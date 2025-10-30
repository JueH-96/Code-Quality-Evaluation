def main():
    import sys
    from collections import defaultdict
    data = sys.stdin.read().strip().split()
    if not data:
        return

    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    M = int(next(it))
    
    operations = []
    for _ in range(M):
        T = int(next(it))
        A = int(next(it))
        X = int(next(it))
        operations.append((T, A, X))
    
    # used_row and used_col keep track of whether a row or col has been "finalized" (updated by a later op in the original order).
    used_row = [False] * (H + 1)
    used_col = [False] * (W + 1)
    count_row = 0  # how many rows are finalized by later ops
    count_col = 0  # how many columns are finalized
    
    # Dictionary to record the total number of cells with a given color.
    color_count = defaultdict(int)
    
    # Process operations in reverse in order to account only for the last op that affected each row or column.
    for T, A, X in reversed(operations):
        if T == 1:  # row operation: repaint entire row A with color X
            if not used_row[A]:
                # This row op affects exactly the cells in that row whose column hasn't already been finalized.
                contribution = W - count_col
                color_count[X] += contribution
                used_row[A] = True
                count_row += 1
        else:  # T == 2: column operation: repaint entire column A with color X
            if not used_col[A]:
                # This column op affects cells in that column for rows that haven't been finalized.
                contribution = H - count_row
                color_count[X] += contribution
                used_col[A] = True
                count_col += 1

    # The remaining cells that were never repainted (in reverse simulation, rows and cols not finalized)
    # still have the initial color 0.
    initial_cells = (H - count_row) * (W - count_col)
    color_count[0] += initial_cells

    # Filter out the colors that have positive counts and sort them by the color value.
    result = [(color, cnt) for color, cnt in color_count.items() if cnt > 0]
    result.sort(key=lambda x: x[0])

    # Prepare and print the output.
    out_lines = []
    out_lines.append(str(len(result)))
    for color, cnt in result:
        out_lines.append(f"{color} {cnt}")
    
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()