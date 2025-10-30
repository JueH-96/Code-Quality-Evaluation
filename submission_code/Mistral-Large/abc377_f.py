import sys

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    positions = [(int(data[2 * i + 2]), int(data[2 * i + 3])) for i in range(M)]

    rows = set()
    cols = set()
    diag1 = set()
    diag2 = set()

    for a, b in positions:
        rows.add(a)
        cols.add(b)
        diag1.add(a + b)
        diag2.add(a - b)

    unsafe_rows = len(rows) * N
    unsafe_cols = len(cols) * N
    unsafe_diag1 = len(diag1) * N
    unsafe_diag2 = len(diag2) * N

    total_unsafe = unsafe_rows + unsafe_cols + unsafe_diag1 + unsafe_diag2 - 3 * M

    safe_squares = N * N - total_unsafe

    print(safe_squares)

if __name__ == "__main__":
    main()