import sys

def main():
    input = sys.stdin.readline
    N = int(input())
    S = [input().rstrip('
') for _ in range(N)]
    M = max(len(s) for s in S)

    # Create an N x M grid filled with '*'
    grid = [['*'] * M for _ in range(N)]

    # Place each S_i into its row (counting from bottom)
    # S_1 goes to bottom row (index N-1), S_2 to N-2, ..., S_N to row 0
    for i, s in enumerate(S):
        row = N - 1 - i
        for j, ch in enumerate(s):
            grid[row][j] = ch

    # For each column j, build T_j by reading top to bottom and
    # stripping trailing '*' so it doesn't end with '*'
    out = []
    for j in range(M):
        col = ''.join(grid[r][j] for r in range(N)).rstrip('*')
        out.append(col)

    # Print the result
    sys.stdout.write('
'.join(out))

if __name__ == "__main__":
    main()