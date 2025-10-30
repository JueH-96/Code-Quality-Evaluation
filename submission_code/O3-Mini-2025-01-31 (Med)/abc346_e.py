def main():
    import sys
    input = sys.stdin.readline

    H, W, M = map(int, input().split())
    # Reading all operations
    ops = [tuple(map(int, input().split())) for _ in range(M)]

    # Boolean markers to indicate whether a row/column has been painted
    row_visited = [False] * (H + 1)
    col_visited = [False] * (W + 1)
    unvisited_rows = H  # number of rows not painted in reverse iteration
    unvisited_cols = W  # number of columns not painted in reverse iteration

    # Dictionary to store the count of cells by color
    ans = {}

    # Process the operations in reverse order.
    # When processing in reverse, an operation is effective only if the row/column hasn't been "visited" yet.
    for t, a, x in reversed(ops):
        if t == 1:
            # Operation on a row
            if not row_visited[a]:
                row_visited[a] = True
                # The effect of this op covers all unvisited columns
                ans[x] = ans.get(x, 0) + unvisited_cols
                unvisited_rows -= 1
        else:
            # Operation on a column (t == 2)
            if not col_visited[a]:
                col_visited[a] = True
                # The effect of this op covers all unvisited rows
                ans[x] = ans.get(x, 0) + unvisited_rows
                unvisited_cols -= 1

    # Finally, the remaining cells (that never got repainted by any effective operation)
    # retain the initial color 0.
    remaining_initial = unvisited_rows * unvisited_cols
    ans[0] = ans.get(0, 0) + remaining_initial

    # Prepare output: Only include colors with more than 0 cells.
    colors = [c for c in ans if ans[c] > 0]
    colors.sort()

    # Print the result according to the required format.
    print(len(colors))
    for c in colors:
        print(c, ans[c])


if __name__ == '__main__':
    main()