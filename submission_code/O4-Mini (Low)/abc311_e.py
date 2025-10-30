import sys
import threading

def main():
    import sys

    input = sys.stdin.readline
    H, W, N = map(int, input().split())
    # Store holes by row for O(1) lookup
    holes_by_row = [set() for _ in range(H+1)]
    for _ in range(N):
        a, b = map(int, input().split())
        holes_by_row[a].add(b)

    # We only need two rows of DP at a time: the "previous" dp row and the "current" one.
    # dp[j]  = size of largest square ending at (current_row, j)
    # prev[j] = size of largest square ending at (current_row-1, j)
    prev_dp = [0] * (W + 1)
    curr_dp = [0] * (W + 1)

    total = 0
    for i in range(1, H+1):
        row_holes = holes_by_row[i]
        # We reset curr_dp[0] = 0 for the "column 0" sentinel
        curr_dp[0] = 0
        for j in range(1, W+1):
            if j in row_holes:
                # A hole forces square size to zero here
                curr_dp[j] = 0
            else:
                # Take min of top (prev_dp[j]), left (curr_dp[j-1]),
                # and top-left (prev_dp[j-1]) + 1
                # This is the classic DP for counting all-1 square submatrices.
                v = prev_dp[j]
                if curr_dp[j-1] < v:
                    v = curr_dp[j-1]
                if prev_dp[j-1] < v:
                    v = prev_dp[j-1]
                curr_dp[j] = v + 1
            total += curr_dp[j]
        # swap dp rows
        prev_dp, curr_dp = curr_dp, prev_dp

    # Output the total count of holeless (all-1) squares
    print(total)

if __name__ == "__main__":
    main()