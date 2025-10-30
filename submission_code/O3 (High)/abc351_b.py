import sys

def main() -> None:
    data = sys.stdin.read().splitlines()
    N = int(data[0])
    A_rows = data[1:1 + N]
    B_rows = data[1 + N:1 + 2 * N]

    for i in range(N):              # 0-based row index
        for j in range(N):          # 0-based column index
            if A_rows[i][j] != B_rows[i][j]:
                print(i + 1, j + 1)  # convert to 1-based
                return

if __name__ == "__main__":
    main()