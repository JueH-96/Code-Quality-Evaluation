import sys

def main() -> None:
    stdin = sys.stdin.readline

    # Read N, Q
    N_Q = stdin().split()
    while len(N_Q) < 2:        # guard against blank lines
        N_Q = stdin().split()
    N, Q = map(int, N_Q)

    # Read initial string and store as list for O(1) updates
    S = list(stdin().strip())

    # Helper to test for "ABC" starting at position p (0-based)
    def is_abc(pos: int) -> bool:
        return (
            0 <= pos <= N - 3 and
            S[pos] == 'A' and
            S[pos + 1] == 'B' and
            S[pos + 2] == 'C'
        )

    # Initial number of occurrences
    abc_cnt = sum(1 for i in range(N - 2) if is_abc(i))

    out_lines = []

    for _ in range(Q):
        x_str, c = stdin().split()
        idx = int(x_str) - 1      # convert to 0-based index

        # Remove contributions of the three possibly affected windows
        for p in (idx - 2, idx - 1, idx):
            if is_abc(p):
                abc_cnt -= 1

        # Perform the character update
        S[idx] = c

        # Add contributions of the same three windows after the change
        for p in (idx - 2, idx - 1, idx):
            if is_abc(p):
                abc_cnt += 1

        out_lines.append(str(abc_cnt))

    sys.stdout.write('
'.join(out_lines))


if __name__ == "__main__":
    main()