import sys

def main():
    input = sys.stdin.readline
    N = int(input())
    grid = [input().rstrip() for _ in range(N)]
    
    # Count 'o's in each row and each column
    row_counts = [row.count('o') for row in grid]
    col_counts = [0] * N
    for row in grid:
        for j, ch in enumerate(row):
            if ch == 'o':
                col_counts[j] += 1

    # For each cell that is 'o', treat it as the "corner" of an L-shape.
    # It contributes (row_o_count - 1) * (col_o_count - 1) triples.
    ans = 0
    for i, row in enumerate(grid):
        ri = row_counts[i] - 1
        if ri <= 0:
            continue
        for j, ch in enumerate(row):
            if ch == 'o':
                cj = col_counts[j] - 1
                if cj > 0:
                    ans += ri * cj

    print(ans)


if __name__ == '__main__':
    main()