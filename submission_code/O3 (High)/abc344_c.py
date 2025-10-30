import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    it = iter(data)

    # read sequences
    N = next(it)
    A = [next(it) for _ in range(N)]
    M = next(it)
    B = [next(it) for _ in range(M)]
    L = next(it)
    C = [next(it) for _ in range(L)]
    Q = next(it)
    X = [next(it) for _ in range(Q)]

    # pre–compute all possible sums of one element from A and one from B
    ab_sums = {a + b for a in A for b in B}

    out_lines = []
    for x in X:
        possible = False
        for c in C:
            if (x - c) in ab_sums:
                possible = True
                break
        out_lines.append("Yes" if possible else "No")

    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()