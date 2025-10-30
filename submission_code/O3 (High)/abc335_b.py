import sys

def main() -> None:
    # Read N
    N_line = sys.stdin.readline().strip()
    if not N_line:
        return
    N = int(N_line)

    # Enumerate triples in lexicographical order
    # x increases fastest in first position of triple, etc.
    # Constraints are tiny (N ≤ 21), so a triple nested loop is fine.
    out_lines = []
    for x in range(N + 1):
        max_y = N - x
        for y in range(max_y + 1):
            max_z = N - x - y
            for z in range(max_z + 1):
                out_lines.append(f"{x} {y} {z}")

    # Output all collected triples
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()