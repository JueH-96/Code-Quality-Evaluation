import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return

    it = iter(data)

    # Number of people
    N = int(next(it))

    # P_1 … P_N : person numbers in order from the front
    positions = {}
    for idx in range(1, N + 1):
        person = int(next(it))
        positions[person] = idx          # person number -> position (1-indexed)

    # Number of queries
    Q = int(next(it))

    out = []
    for _ in range(Q):
        A = int(next(it))
        B = int(next(it))

        # Compare their positions: smaller index ⇒ closer to the front
        if positions[A] < positions[B]:
            out.append(str(A))
        else:
            out.append(str(B))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()