import sys


def min_operations(n: int, k: int, s: str) -> int:
    """
    Returns the minimal number of operations that turn every black cell white
    by whitening any k consecutive cells in one operation.
    """
    # collect indices of black cells
    blacks = [i for i, c in enumerate(s) if c == 'B']
    if not blacks:                         # already all white
        return 0

    res = 0                               # number of operations
    ptr = 0                               # pointer inside `blacks`

    while ptr < len(blacks):
        pos = blacks[ptr]                 # left-most still un-covered black

        # Can we start a segment at `pos` that fits completely inside the strip?
        if pos + k - 1 <= n - 1:
            cover_end = pos + k - 1       # rightmost cell whitened by this move
            res += 1
            # skip all black cells that are inside this whitened segment
            while ptr < len(blacks) and blacks[ptr] <= cover_end:
                ptr += 1
        else:
            # `pos` lies in the last k cells of the strip,                     #
            # any valid segment that whitens `pos` must be the suffix          #
            # [n-k, n-1]. This single operation also covers all remaining      #
            # black cells (they are all ≥ pos).                                #
            res += 1
            break

    return res


def main() -> None:
    data = sys.stdin.read().split()
    t = int(data[0])
    out = []
    idx = 1
    for _ in range(t):
        n = int(data[idx])
        k = int(data[idx + 1])
        idx += 2
        s = data[idx]
        idx += 1
        out.append(str(min_operations(n, k, s)))
    sys.stdout.write('
'.join(out))


if __name__ == "__main__":
    main()