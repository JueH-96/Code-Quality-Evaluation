import sys

def main() -> None:
    N_line = sys.stdin.readline()
    if not N_line:
        return
    N = int(N_line.strip())

    result_rows = []
    for r in range(1, N + 1):           # 1-indexed row
        row_chars = []
        for c in range(1, N + 1):       # 1-indexed column
            layer = min(r, c, N + 1 - r, N + 1 - c)
            row_chars.append('#' if layer % 2 == 1 else '.')
        result_rows.append(''.join(row_chars))

    sys.stdout.write('
'.join(result_rows))

if __name__ == "__main__":
    main()