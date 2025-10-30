import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    N, L, R = data[:3]
    A = data[3:3 + N]

    # For each A_i, choose the closest integer inside [L, R].
    # Since A_i is an integer, the closest value is simply clamping A_i to the range.
    res = []
    for a in A:
        if a < L:
            res.append(str(L))
        elif a > R:
            res.append(str(R))
        else:
            res.append(str(a))

    sys.stdout.write(' '.join(res))

if __name__ == "__main__":
    main()