import sys

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    N, R, C = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # Current cumulative displacement after processing i steps (0 at start)
    r, c = 0, 0

    # Set of cumulative vectors for every time an injection happened (starts with time 0)
    seen = {(0, 0)}

    out = []

    for ch in S:
        # 1. wind moves all smoke (update cumulative displacement)
        if ch == 'N':
            r -= 1
        elif ch == 'S':
            r += 1
        elif ch == 'E':
            c += 1
        else:  # 'W'
            c -= 1

        # 2. inject new smoke at origin if it is empty
        if (r, c) not in seen:
            seen.add((r, c))

        # 3. is there smoke at (R, C) now?  -> need some previous injection point (ri,ci)
        #    such that (r - ri, c - ci) == (R, C)  <=>  (ri,ci) == (r-R, c-C)
        if (r - R, c - C) in seen:
            out.append('1')
        else:
            out.append('0')

    print(''.join(out))


if __name__ == "__main__":
    main()