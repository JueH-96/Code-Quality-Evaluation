import sys

def main() -> None:
    input = sys.stdin.readline

    # Read input
    N = int(input().strip())
    S = [input().rstrip('
') for _ in range(N)]

    # Reverse the rows so that S_N becomes the top row
    rows = S[::-1]

    # Maximum length among the strings
    M = max(len(s) for s in S)

    # Build and output each vertical string T_j
    for j in range(M):
        column = []
        for r in range(N):
            if j < len(rows[r]):
                column.append(rows[r][j])
            else:
                column.append('*')

        # Remove trailing '*' so that T_j does not end with '*'
        while column and column[-1] == '*':
            column.pop()

        print(''.join(column))

if __name__ == "__main__":
    main()