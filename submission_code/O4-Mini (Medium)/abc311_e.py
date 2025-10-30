import sys
import threading

def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    if not line:
        return
    H, W, N = map(int, line)
    # Create a 2D boolean grid of holes using bytearrays for space efficiency
    # We pad columns by 2 to avoid bounds checks on j+1.
    holes = [bytearray(W+2) for _ in range(H+2)]
    for _ in range(N):
        a, b = map(int, data.readline().split())
        holes[a][b] = 1

    # dp_curr[j]: dp value for current row i at column j
    # dp_next[j]: dp value for row i+1 at column j
    dp_curr = [0] * (W+2)
    dp_next = [0] * (W+2)

    total = 0
    # Process rows from bottom to top
    for i in range(H, 0, -1):
        # For each column from right to left
        # dp_curr[W+1] is boundary = 0
        for j in range(W, 0, -1):
            if holes[i][j]:
                # a hole => no square of any positive size
                dp_curr[j] = 0
            else:
                # can form a 1x1 square, plus possibly extend by
                # the minimum of right, down, and down-right neighbors
                # which are dp_curr[j+1], dp_next[j], dp_next[j+1]
                # This gives largest square with top-left at (i,j)
                # without holes.
                m = dp_curr[j+1]
                if dp_next[j] < m:
                    m = dp_next[j]
                temp = dp_next[j+1]
                if temp < m:
                    m = temp
                dp_curr[j] = m + 1
                total += dp_curr[j]
        # swap current and next rows for next iteration
        dp_curr, dp_next = dp_next, dp_curr

    # Print the total count of holeless squares
    sys.stdout.write(str(total))

if __name__ == "__main__":
    main()